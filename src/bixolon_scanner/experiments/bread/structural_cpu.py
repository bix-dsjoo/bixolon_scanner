"""Paired build-host CPU scans with fixed affinity and alternating profile order."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import psutil

from ...configuration import load_json_config
from ...contracts.catalog import sha256_file
from ...evaluation.structural_benchmark import percentiles
from ...evaluation.three_bakery import score_response
from ...runtime.imaging import decode_image
from ...runtime.inference_timing import collect_inference_timings
from ...training.three_bakery_data import read_jsonl, write_json
from ...worker.runtime_factory import build_worker_runtime
from ...worker.settings import WorkerSettings


def run(config_path: Path, candidate: Path, output: Path):
    config = load_json_config(config_path)
    process = psutil.Process()
    affinity = process.cpu_affinity()
    process.cpu_affinity(affinity[:4])
    records = read_jsonl(
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
    )
    profiles = {
        "reference": (Path(config["runtime"]), Path(config["catalog"])),
        "student": (candidate / "runtime", candidate / "catalog"),
    }
    runtimes, inputs, rows = {}, [], []
    try:
        for name, (runtime, catalog) in profiles.items():
            runtimes[name] = build_worker_runtime(
                WorkerSettings(
                    package_dir=runtime,
                    catalog_dir=catalog,
                    provider="cpu",
                    cpu_detector_intra_op_threads=4,
                    cpu_embedder_intra_op_threads=4,
                    reuse_verifier_embeddings=True,
                )
            )
        for record in records:
            path = Path(record["image_path"])
            if sha256_file(path) != record["image_sha256"]:
                raise ValueError("paired CPU input checksum mismatch")
            with decode_image(
                path.read_bytes(),
                max_bytes=30000000,
                max_pixels=50000000,
                jpeg_draft_size=runtimes["reference"].jpeg_draft_size,
            ) as image:
                inputs.append(image.copy())
        for index in range(10):
            for name, runtime in runtimes.items():
                runtime.scan(inputs[index], f"warm-{name}-{index}")
        for repeat in range(2):
            for index, (record, image) in enumerate(zip(records, inputs, strict=True)):
                order = (
                    ["reference", "student"]
                    if (index + repeat) % 2 == 0
                    else ["student", "reference"]
                )
                for name in order:
                    with collect_inference_timings() as calls:
                        start = time.perf_counter()
                        response = runtimes[name].scan(image, f"paired-{repeat}-{name}-{index}")
                        elapsed = (time.perf_counter() - start) * 1000
                    rows.append(
                        {
                            "profile": name,
                            "repetition": repeat,
                            "image_id": record["image_id"],
                            "elapsed_ms": elapsed,
                            "response": response.model_dump(mode="json"),
                            "model_calls": calls,
                            "metrics": score_response(
                                response, record["annotations"], threshold=0.5
                            ),
                        }
                    )
                if (index + 1) % 33 == 0:
                    print(f"paired CPU {repeat + 1}: {index + 1}/{len(records)}", flush=True)
    finally:
        for image in inputs:
            image.close()
        for runtime in runtimes.values():
            runtime.close()
        process.cpu_affinity(affinity)
    write_json(
        output,
        {
            "scope": "build host, four logical CPUs, alternating AB/BA, 2 repetitions; excludes HTTP/decode; not N100",
            "affinity": affinity[:4],
            "rows": rows,
            "latency": {
                name: percentiles([r["elapsed_ms"] for r in rows if r["profile"] == name])
                for name in profiles
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.candidate, args.output)


if __name__ == "__main__":
    main()
