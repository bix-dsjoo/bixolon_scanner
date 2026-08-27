from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from ..contracts.catalog import sha256_file


def optimize_fixed_onnx(
    source_path: Path,
    source_report_path: Path,
    output_path: Path,
) -> dict:
    """Constant-fold a fixed-shape ONNX graph without hardware-specific rewrites."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if report.get("onnx_sha256") != sha256_file(source_path):
        raise ValueError("source ONNX and export report checksums differ")
    if not isinstance(report.get("fixed_batch_size"), int):
        raise ValueError("portable ONNX optimization requires a fixed batch size")

    import onnxruntime as ort

    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(output_path)
    options.log_severity_level = 3
    ort.InferenceSession(
        str(source_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if not output_path.is_file():
        raise RuntimeError("ONNX Runtime did not emit the optimized model")
    optimized = dict(report)
    optimized.update(
        {
            "operation": "optimize_fixed_batch_onnx",
            "origin_onnx_sha256": report.get(
                "origin_onnx_sha256", report.get("source_onnx_sha256")
            ),
            "source_onnx": source_path.resolve().as_posix(),
            "source_report": source_report_path.resolve().as_posix(),
            "source_report_sha256": sha256_file(source_report_path),
            "output_onnx": output_path.resolve().as_posix(),
            "onnx_sha256": sha256_file(output_path),
            "source_onnx_sha256": sha256_file(source_path),
            "optimization": "ORT_ENABLE_EXTENDED",
            "optimization_provider": "CPUExecutionProvider",
            "onnxruntime_version": ort.__version__,
            "optimization_platform": platform.platform(),
            "hardware_specific_optimization": False,
            "weights_modified": False,
        }
    )
    return optimized


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optimize a fixed-shape ONNX model")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = optimize_fixed_onnx(args.source, args.source_report, args.output)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
