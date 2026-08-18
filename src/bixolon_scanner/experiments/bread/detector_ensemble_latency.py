from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.onnx import OrtRunner


def latency_summary(values: list[float]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    if not len(samples):
        raise ValueError("latency summary requires samples")
    return {
        "sample_count": len(values),
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "minimum_ms": float(samples.min()),
        "maximum_ms": float(samples.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runners = [OrtRunner(path, "cuda", args.cuda_dll_dir) for path in args.models]
    inputs = [runner.session.get_inputs() for runner in runners]
    outputs = [runner.session.get_outputs() for runner in runners]
    contracts = [
        (
            [(item.name, item.shape, item.type) for item in session_inputs],
            [(item.name, item.shape, item.type) for item in session_outputs],
        )
        for session_inputs, session_outputs in zip(inputs, outputs)
    ]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("ensemble ONNX contracts differ")
    input_name = inputs[0][0].name
    input_shape = inputs[0][0].shape
    if len(input_shape) != 4 or not all(isinstance(value, int) for value in input_shape[1:]):
        raise ValueError("ensemble latency probe requires a fixed CHW detector input")
    output_names = [item.name for item in outputs[0]]
    tensor = np.zeros((1, *input_shape[1:]), dtype=np.float32)

    for _ in range(args.warmup):
        for runner in runners:
            runner.run(output_names, input_name, tensor)

    sequential_latencies = []
    per_model_latencies: list[list[float]] = [[] for _ in runners]
    for _ in range(args.runs):
        ensemble_started = time.perf_counter()
        for index, runner in enumerate(runners):
            model_started = time.perf_counter()
            runner.run(output_names, input_name, tensor)
            per_model_latencies[index].append((time.perf_counter() - model_started) * 1000.0)
        sequential_latencies.append((time.perf_counter() - ensemble_started) * 1000.0)

    parallel_latencies = []
    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        for _ in range(args.parallel_warmup):
            futures = [
                executor.submit(runner.run, output_names, input_name, tensor) for runner in runners
            ]
            for future in futures:
                future.result()
        for _ in range(args.runs):
            started = time.perf_counter()
            futures = [
                executor.submit(runner.run, output_names, input_name, tensor) for runner in runners
            ]
            for future in futures:
                future.result()
            parallel_latencies.append((time.perf_counter() - started) * 1000.0)

    import onnxruntime as ort

    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_fixed_ensemble_cuda_latency",
        "latency_scope": (
            "four sequential CUDA ONNX forwards with CPU input binding and output copy; "
            "decode, resize, fusion, classifier, and API excluded"
        ),
        "platform": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "provider": "CUDAExecutionProvider",
        "warmup_count_per_model": args.warmup,
        "parallel_warmup_count": args.parallel_warmup,
        "run_count": args.runs,
        "input_shape": [1, *input_shape[1:]],
        "models": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "latency": latency_summary(values),
            }
            for path, values in zip(args.models, per_model_latencies)
        ],
        "sequential_ensemble_latency": latency_summary(sequential_latencies),
        "parallel_ensemble_latency": latency_summary(parallel_latencies),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a fixed detector ensemble with CUDA ONNX Runtime"
    )
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--cuda-dll-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--parallel-warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
