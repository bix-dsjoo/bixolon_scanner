from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .onnx_int8_probe import _embedding_parity, _sha256, load_calibration_tensors


def run(args: argparse.Namespace) -> dict[str, Any]:
    import nncf
    import onnx

    tensors = load_calibration_tensors(
        args.manifest,
        args.dataset_root,
        input_size=args.input_size,
        maximum_samples=args.maximum_samples,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )
    calibration_dataset = nncf.Dataset(
        tensors,
        transform_func=lambda tensor: {args.input_name: tensor},
    )
    source = onnx.load(args.source_model)
    presets = {
        "mixed": nncf.QuantizationPreset.MIXED,
        "performance": nncf.QuantizationPreset.PERFORMANCE,
    }
    target_devices = {
        "cpu": nncf.TargetDevice.CPU,
        "gpu": nncf.TargetDevice.GPU,
    }
    advanced = nncf.AdvancedQuantizationParameters(
        smooth_quant_alpha=args.smooth_quant_alpha,
    )
    candidate = nncf.quantize(
        source,
        calibration_dataset,
        preset=presets[args.preset],
        target_device=target_devices[args.target_device],
        subset_size=len(tensors),
        model_type=nncf.ModelType.TRANSFORMER if args.transformer else None,
        advanced_parameters=advanced,
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, args.output_model)
    report = {
        "schema_version": "1.0",
        "evaluation": "nncf_onnx_int8_embedding_probe",
        "nncf_version": nncf.__version__,
        "source_model": str(args.source_model.resolve()),
        "source_model_sha256": _sha256(args.source_model),
        "source_size_bytes": args.source_model.stat().st_size,
        "candidate_model": str(args.output_model.resolve()),
        "candidate_model_sha256": _sha256(args.output_model),
        "candidate_size_bytes": args.output_model.stat().st_size,
        "calibration": {
            "manifest_sha256": _sha256(args.manifest),
            "sample_count": len(tensors),
            "input_size": args.input_size,
            "preset": args.preset,
            "target_device": args.target_device,
            "transformer": bool(args.transformer),
            "smooth_quant_alpha": args.smooth_quant_alpha,
        },
        "embedding_parity": _embedding_parity(
            args.source_model,
            args.output_model,
            input_name=args.input_name,
            output_name=args.output_name,
            tensors=tensors,
            sample_count=args.parity_samples,
        ),
        "limitations": [
            "calibration uses only the locked single-object Catalog source",
            "full pipeline accuracy and target-device latency are evaluated separately",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize an ONNX embedder with NNCF INT8 PTQ")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--output-name", default="embeddings")
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--maximum-samples", type=int, default=240)
    parser.add_argument("--parity-samples", type=int, default=24)
    parser.add_argument("--preset", choices=("mixed", "performance"), default="mixed")
    parser.add_argument("--target-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--smooth-quant-alpha", type=float)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    parser.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
