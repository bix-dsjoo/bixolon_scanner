"""Freeze and compare actual 0.1.17/0.1.18 Worker EXEs on the fixed regression set."""

from __future__ import annotations

import argparse
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.three_bakery_http import final_records, measure, provider_parity
from bixolon_scanner.training.three_bakery_data import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/versions/0.1.18/final-evaluation"
    config_path = root / "configs/experiments/bread/three_bakery_revised300.json"
    model_selection = (
        root / "artifacts/retraining/three-bakery-improvement-0.1.18/release-selection.json"
    )
    selection = load_json_config(model_selection)
    if selection["model_changed"] or selection["policy_changed"]:
        raise ValueError("this evaluation is for the retained model and CPU-only execution change")
    candidates = {v: root / f"artifacts/versions/{v}/staging" for v in ("0.1.17", "0.1.18")}
    workers = {
        (v, p): root
        / (
            f"artifacts/installers/{v}/windows-payload/worker/bixolon-worker.exe"
            if p == "cpu"
            else f"artifacts/versions/{v}/bixolon-bakery-ai-scanner-{v}/worker/bixolon-worker.exe"
        )
        for v in candidates
        for p in ("cpu", "cuda")
    }
    interpreters = {
        "cpu": root / "artifacts/build-envs/worker-cpu-0.1.17-py311/Scripts/python.exe",
        "cuda": root / "artifacts/build-envs/worker-cuda-py311/Scripts/python.exe",
    }
    freeze = {
        "version": "0.1.18",
        "selection_sha256": sha256_file(model_selection),
        "config_sha256": sha256_file(config_path),
        "version_config_sha256": sha256_file(root / "configs/versions/0.1.18.json"),
        "cpu_profile": [8, 12],
        "repetitions": 3,
        "warmup": 10,
        "candidates": {
            v: {
                n: directory_content_manifest(p / n)["manifest_sha256"]
                for n in ("runtime", "catalog")
            }
            for v, p in candidates.items()
        },
        "workers": {
            f"{v}/{p}": directory_content_manifest(w.parent)["manifest_sha256"]
            for (v, p), w in workers.items()
        },
        "code_sha256": {
            p.relative_to(root).as_posix(): sha256_file(p)
            for d in ("contracts", "pipeline", "runtime", "worker", "evaluation")
            for p in sorted((root / "src/bixolon_scanner" / d).glob("*.py"))
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "acceptance": "identical statuses/class ranks/correct and wrong approval counts; no CPU all/full p95 regression",
        "scope": "historically exposed fixed regression benchmark; no training or retuning after access",
    }
    freeze_path = output / "final-candidate.json"
    if freeze_path.exists() and load_json_config(freeze_path) != freeze:
        raise ValueError("frozen release, policy, Worker or evaluation code changed")
    write_json(freeze_path, freeze)
    if args.freeze_only:
        print(f"Frozen release and evaluation: {freeze_path}")
        return
    records = final_records(load_json_config(config_path), freeze_path, output)
    identity = {
        "freeze_sha256": sha256_file(freeze_path),
        "input_sha256": sha256_file(output / "final-inputs.jsonl"),
    }
    reports = {}
    for provider in ("cpu", "cuda"):
        for version in candidates:
            report = measure(
                candidates[version],
                records,
                output / version / provider,
                python=interpreters[provider],
                provider=provider,
                cuda_dll_dir=candidates[version] / "cuda-runtime" if provider == "cuda" else None,
                warmup=10,
                repetitions=3,
                match_iou=0.5,
                input_identity=identity,
                cpu_profile=(8, 12),
                worker_executable=workers[version, provider],
                maximum_p95_ms=300 if provider == "cpu" else None,
            )
            reports[f"{version}/{provider}"] = report
    comparisons = {}
    for provider in ("cpu", "cuda"):
        parity = provider_parity(
            output / "0.1.17" / provider / "responses.jsonl",
            output / "0.1.18" / provider / "responses.jsonl",
        )
        old, new = reports[f"0.1.17/{provider}"], reports[f"0.1.18/{provider}"]
        comparisons[provider] = {
            "parity": parity,
            "p95_ratios": {
                g: [
                    n["latency"][g]["p95_ms"] / o["latency"][g]["p95_ms"]
                    for o, n in zip(old["repetitions"], new["repetitions"], strict=True)
                ]
                for g in ("all", "full_path")
            },
            "accuracy_equal": all(
                all(
                    o[k] == n[k]
                    for k in (
                        "correct_approved_count",
                        "wrong_approved_count",
                        "ground_truth_count",
                        "image_count",
                        "missed_count",
                    )
                )
                for o, n in zip(old["repetitions"], new["repetitions"], strict=True)
            ),
        }
    parity = provider_parity(
        output / "0.1.18/cpu/responses.jsonl", output / "0.1.18/cuda/responses.jsonl"
    )
    result = {
        "reports": {
            k: {"repetitions": v["repetitions"], "environment": v["environment"]}
            for k, v in reports.items()
        },
        "comparison": comparisons,
        "cpu_cuda_parity": parity,
        "regression_passed": all(
            v["parity"]["status_rank_parity"] and v["accuracy_equal"] for v in comparisons.values()
        )
        and parity["status_rank_parity"]
        and all(r <= 1.05 for values in comparisons["cpu"]["p95_ratios"].values() for r in values),
        "claim_scope": freeze["scope"],
    }
    write_json(output / "report.json", result)
    print({"regression_passed": result["regression_passed"], "comparison": comparisons}, flush=True)


if __name__ == "__main__":
    main()
