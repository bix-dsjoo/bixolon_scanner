"""Same-tensor structural model timing. Synthetic inputs are not accuracy evidence."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..training.three_bakery_data import write_json


def percentiles(values: list[float]) -> dict:
    return {
        "count": len(values),
        **{f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)},
        "maximum_ms": max(values),
    }


def benchmark(path: Path, config: dict, *, batches: list[int], provider: str = "cpu") -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = config["cpu_threads"]
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if provider == "openvino"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    providers = (
        ["CPUExecutionProvider"]
        if provider == "cpu"
        else [("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"})]
    )
    start = time.perf_counter()
    session = ort.InferenceSession(str(path), options, providers=providers)
    if provider == "openvino" and session.get_providers()[0] != "OpenVINOExecutionProvider":
        raise RuntimeError("requested GPU provider was not initialized")
    compile_ms = (time.perf_counter() - start) * 1000
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("tensor benchmark needs a single image input")
    rows = []
    for size in batches:
        shape = [size, *inputs[0].shape[1:]]
        if not all(isinstance(v, int) for v in shape):
            raise ValueError("only the batch dimension may be dynamic")
        if isinstance(inputs[0].shape[0], int) and size != inputs[0].shape[0]:
            raise ValueError("benchmark batch differs from fixed graph")
        values = np.random.default_rng(910 + size).normal(size=shape).astype(np.float32)
        for _ in range(config["warmup"]):
            session.run(None, {inputs[0].name: values})
        samples = []
        for _ in range(config["repetitions"]):
            start = time.perf_counter()
            session.run(None, {inputs[0].name: values})
            samples.append((time.perf_counter() - start) * 1000)
        rows.append({"batch": size, "latency": percentiles(samples), "samples_ms": samples})
    return {
        "model_sha256": sha256_file(path),
        "provider": session.get_providers(),
        "ort": ort.__version__,
        "compile_ms": compile_ms,
        "rows": rows,
        "host": platform.platform(),
        "scope": "build host; synthetic same-tensor benchmark; excludes HTTP; not N100 or accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", action="store_true")
    args = parser.parse_args()
    settings = load_json_config(args.config)["benchmark"]
    write_json(
        args.output,
        benchmark(args.model, settings, batches=[1] if args.detector else settings["batches"]),
    )


if __name__ == "__main__":
    main()
