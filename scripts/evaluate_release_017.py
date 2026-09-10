"""Evaluate the user-frozen 0.1.17 packaged Workers on the final fixed benchmark."""

import argparse
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.three_bakery_http import (
    final_records,
    interpreter_identity,
    measure,
    provider_parity,
)
from bixolon_scanner.evaluation.three_bakery_objective import (
    apply_final_objective,
    load_objective,
)
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/versions/0.1.17/final-evaluation"
    version_root = output.parent
    candidate = version_root / "staging"
    cpu_worker = root / "artifacts/installers/0.1.17/windows-payload/worker/bixolon-worker.exe"
    cuda_worker = version_root / "bixolon-bakery-ai-scanner-0.1.17/worker/bixolon-worker.exe"
    environments = {
        "cpu": root / "artifacts/build-envs/worker-cpu-0.1.17-py311/Scripts/python.exe",
        "cuda": root / "artifacts/build-envs/worker-cuda-py311/Scripts/python.exe",
    }
    objective_path = root / "configs/experiments/bread/three_bakery_objective.json"
    config_path = root / "configs/experiments/bread/three_bakery_revised300.json"
    freeze = {
        "product_version": "0.1.17",
        "candidate_id": "ssdlite-margin-dense-20260908",
        "selection": "user_requested_current_first_place_before_final_access",
        "runtime": directory_content_manifest(candidate / "runtime")["manifest_sha256"],
        "catalog": directory_content_manifest(candidate / "catalog")["manifest_sha256"],
        "cpu_profile": [8, 12],
        "evaluation_config_sha256": sha256_file(config_path),
        "objective_sha256": sha256_file(objective_path),
        "version_config_sha256": sha256_file(root / "configs/versions/0.1.17.json"),
        "code_sha256": {
            p.relative_to(root).as_posix(): sha256_file(p)
            for directory in ("contracts", "pipeline", "runtime", "worker", "evaluation")
            for p in sorted((root / "src/bixolon_scanner" / directory).glob("*.py"))
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "packaged_worker_sha256": {
            "cpu": directory_content_manifest(cpu_worker.parent)["manifest_sha256"],
            "cuda": directory_content_manifest(cuda_worker.parent)["manifest_sha256"],
        },
        "interpreters": {k: interpreter_identity(v) for k, v in environments.items()},
    }
    freeze_path = output / "final-candidate.json"
    if freeze_path.exists() and load_json_config(freeze_path) != freeze:
        raise ValueError("The frozen release or evaluation code changed")
    write_json(freeze_path, freeze)
    if args.freeze_only:
        print(f"Release and evaluation hashes frozen: {freeze_path}")
        return
    config = load_json_config(config_path)
    records = final_records(config, freeze_path, output)
    sources = read_jsonl(
        root / "artifacts/retraining/three-bakery-revised300-roi-integrity/sources/originals.jsonl"
    )
    if {r["image_sha256"] for r in records} & {r["image_sha256"] for r in sources}:
        raise ValueError("Final benchmark duplicates source originals")
    identity = {
        "freeze_sha256": sha256_file(freeze_path),
        "manifest_sha256": sha256_file(output / "final-inputs.jsonl"),
    }
    results = {}
    for provider, executable in (("cpu", cpu_worker), ("cuda", cuda_worker)):
        report = measure(
            candidate,
            records,
            output / provider,
            python=environments[provider],
            provider=provider,
            cuda_dll_dir=candidate / "cuda-runtime" if provider == "cuda" else None,
            warmup=10,
            repetitions=3,
            match_iou=0.5,
            input_identity=identity,
            cpu_profile=(8, 12),
            maximum_p95_ms=300.0 if provider == "cpu" else None,
            worker_executable=executable,
        )
        results[provider] = apply_final_objective(report, load_objective(objective_path))
        write_json(output / provider / "objective-report.json", results[provider])
    write_json(
        output / "report.json",
        {
            "providers": results,
            "provider_parity": provider_parity(
                output / "cpu/responses.jsonl", output / "cuda/responses.jsonl"
            ),
            "target_met": all(r["target_met_all_repetitions"] for r in results.values()),
            "claim_scope": "observed_results_on_historically_used_fixed_300_image_benchmark",
            "freeze_sha256": sha256_file(freeze_path),
        },
    )


if __name__ == "__main__":
    main()
