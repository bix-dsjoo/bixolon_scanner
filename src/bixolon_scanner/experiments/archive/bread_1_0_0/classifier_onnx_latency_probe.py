from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ....runtime.onnx import OrtRunner


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    tensors = np.load(args.tensors, mmap_mode="r")
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    reports = []
    for batch_size in args.batch_sizes:
        batch = np.asarray(tensors[:batch_size], dtype=np.float32)
        for _ in range(args.warmup):
            runner.run(["logits"], "pixel_values", batch)
        durations = []
        for _ in range(args.iterations):
            started = perf_counter()
            runner.run(["logits"], "pixel_values", batch)
            durations.append((perf_counter() - started) * 1000.0)
        reports.append(
            {
                "batch_size": batch_size,
                "sample_count": len(durations),
                "mean_ms": float(np.mean(durations)),
                "p50_ms": float(np.percentile(durations, 50)),
                "p95_ms": float(np.percentile(durations, 95)),
                "p99_ms": float(np.percentile(durations, 99)),
            }
        )
    report = {
        "schema_version": "1.0",
        "evaluation": "classifier_onnx_dynamic_batch_latency_probe",
        "provider": args.provider,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "batches": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark classifier ONNX dynamic batches")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 27, 40])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
