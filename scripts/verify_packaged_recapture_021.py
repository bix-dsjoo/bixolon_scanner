"""Validate actual packaged Workers, then compare paired log132 HTTP results."""

import argparse
from pathlib import Path

from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.evaluation.recapture_improvement import (
    compare_repetitions,
    retained_prediction_parity,
)
from bixolon_scanner.evaluation.scanner_v2_packaged_worker_smoke import main as smoke
from bixolon_scanner.evaluation.three_bakery_http import measure, provider_parity
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def worker(version, provider):
    return Path(
        f"artifacts/installers/{version}/windows-payload/worker"
        if provider == "cpu"
        else f"artifacts/versions/{version}/bixolon-bakery-ai-scanner-{version}/worker"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    records = read_jsonl(
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
    )
    output = Path("artifacts/retraining/recapture-0.2.1/packaged")
    for provider in ("cpu", "cuda"):
        package = worker("0.2.1", provider)
        smoke_args = [
            "--worker-artifact",
            str(package),
            "--executable",
            str(package / "bixolon-worker.exe"),
            "--runtime",
            str(package / "model-package"),
            "--catalog",
            str(package / "store-catalog"),
            "--image",
            records[0]["image_path"],
            "--store-id",
            "three_bakery",
            "--provider",
            provider,
            "--expected-version",
            "0.2.1",
            "--requirements-lock",
            f"configs/runtime/requirements-windows-{provider}.lock",
            "--output",
            f"artifacts/versions/0.2.1/packaged-{provider}-smoke.json",
        ]
        if provider == "cuda":
            smoke_args += ["--cuda-dll-dir", "artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64"]
        smoke(smoke_args)
        if args.smoke_only:
            continue
        reports = {}
        for version in ("0.2.0", "0.2.1"):
            package = worker(version, provider)
            staging = Path(f"artifacts/versions/{version}/staging")
            for source, target in (("runtime", "model-package"), ("catalog", "store-catalog")):
                assert directory_content_manifest(staging / source) == directory_content_manifest(
                    package / target
                )
            result = measure(
                staging,
                records,
                output / provider / version,
                python=Path("artifacts/build-envs")
                / ("worker-cpu-0.1.17-py311" if provider == "cpu" else "worker-cuda-py311")
                / "Scripts/python.exe",
                provider=provider,
                cpu_profile=(8, 12),
                warmup=10,
                repetitions=1,
                worker_executable=package / "bixolon-worker.exe",
                cuda_dll_dir=Path("artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64")
                if provider == "cuda"
                else None,
                input_identity={
                    "scope": "Actual packaged EXE; staging Runtime/Catalog verified byte-identical to included package"
                },
            )
            reports[version] = result["summary"]
            print(provider, version, result["summary"], flush=True)
        comparison = compare_repetitions([reports["0.2.0"]], [reports["0.2.1"]])
        comparison["retained"] = retained_prediction_parity(
            read_jsonl(output / provider / "0.2.0/responses.jsonl"),
            read_jsonl(output / provider / "0.2.1/responses.jsonl"),
        )
        comparison["accepted"] &= comparison["retained"]["unchanged_retained_decisions"]
        write_json(output / provider / "comparison.json", comparison)
    if not args.smoke_only:
        write_json(
            output / "cpu-cuda-parity.json",
            provider_parity(
                output / "cpu/0.2.1/responses.jsonl", output / "cuda/0.2.1/responses.jsonl"
            ),
        )


if __name__ == "__main__":
    main()
