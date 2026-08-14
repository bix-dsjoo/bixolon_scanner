from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ....runtime.onnx import OrtRunner


def latency_summary(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": len(samples),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    tensor = np.zeros((1, 3, args.input_height, args.input_width), dtype=np.float32)
    for _ in range(args.warmup):
        runner.run([args.logits_output, args.boxes_output], args.input_name, tensor)

    durations = []
    for _ in range(args.iterations):
        started = perf_counter()
        runner.run([args.logits_output, args.boxes_output], args.input_name, tensor)
        durations.append((perf_counter() - started) * 1000.0)

    report = {
        "schema_version": "1.0",
        "evaluation": "detector_onnx_raw_latency_probe",
        "model": args.model.name,
        "provider": args.provider,
        "input_size": [args.input_height, args.input_width],
        "warmup": args.warmup,
        "latency_ms": latency_summary(durations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark raw detector ONNX latency")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
