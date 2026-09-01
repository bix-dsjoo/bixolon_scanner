from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...runtime.onnx import prepare_rgb, sigmoid
from .onnx_int8_probe import _sha256


def _sample_rows(rows: list[dict], maximum_samples: int) -> list[dict]:
    if maximum_samples < 1:
        raise ValueError("INT8 calibration sample count must be positive")
    if len(rows) <= maximum_samples:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum_samples, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def load_detector_calibration_tensors(
    manifest: Path,
    dataset_root: Path,
    *,
    input_size: int,
    maximum_samples: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> list[np.ndarray]:
    if input_size < 128:
        raise ValueError("detector INT8 calibration input size must be at least 128")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("detector INT8 calibration manifest is empty")
    root = dataset_root.resolve()
    tensors = []
    for row in _sample_rows(rows, maximum_samples):
        image_path = (root / str(row["image_path"])).resolve()
        if root not in image_path.parents:
            raise ValueError("detector calibration image resolves outside the dataset root")
        if not image_path.is_file():
            raise FileNotFoundError(f"detector calibration image is missing: {image_path}")
        expected_sha256 = row.get("image_sha256")
        if expected_sha256 and _sha256(image_path) != expected_sha256:
            raise ValueError("detector calibration image checksum mismatch")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = prepare_rgb(
                image,
                (input_size, input_size),
                mean,
                std,
                reducing_gap=1.0,
            )
        tensors.append(np.asarray(tensor[None], dtype=np.float32))
    return tensors


def _detector_parity(
    source_model: Path,
    candidate_model: Path,
    *,
    input_name: str,
    logits_output: str,
    boxes_output: str,
    tensors: list[np.ndarray],
    sample_count: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    source = ort.InferenceSession(
        str(source_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    candidate = ort.InferenceSession(
        str(candidate_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    logit_deltas = []
    probability_deltas = []
    box_deltas = []
    top_class_agreements = []
    top_query_agreements = []
    for tensor in tensors[:sample_count]:
        expected_logits, expected_boxes = source.run(
            [logits_output, boxes_output], {input_name: tensor}
        )
        actual_logits, actual_boxes = candidate.run(
            [logits_output, boxes_output], {input_name: tensor}
        )
        expected_logits = np.asarray(expected_logits, dtype=np.float64)
        actual_logits = np.asarray(actual_logits, dtype=np.float64)
        expected_boxes = np.asarray(expected_boxes, dtype=np.float64)
        actual_boxes = np.asarray(actual_boxes, dtype=np.float64)
        if (
            expected_logits.shape != actual_logits.shape
            or expected_boxes.shape != actual_boxes.shape
        ):
            raise ValueError("source and candidate detector output shapes differ")
        expected_probabilities = sigmoid(expected_logits)
        actual_probabilities = sigmoid(actual_logits)
        logit_deltas.append(np.abs(expected_logits - actual_logits).reshape(-1))
        probability_deltas.append(np.abs(expected_probabilities - actual_probabilities).reshape(-1))
        box_deltas.append(np.abs(expected_boxes - actual_boxes).reshape(-1))
        top_class_agreements.append(
            np.equal(
                np.argmax(expected_logits, axis=-1),
                np.argmax(actual_logits, axis=-1),
            ).reshape(-1)
        )
        expected_scores = expected_probabilities.max(axis=-1)
        actual_scores = actual_probabilities.max(axis=-1)
        top_query_agreements.append(
            np.asarray(
                [int(np.argmax(expected_scores)) == int(np.argmax(actual_scores))],
                dtype=bool,
            )
        )

    def delta_summary(parts: list[np.ndarray]) -> dict[str, float]:
        values = np.concatenate(parts)
        return {
            "mean_absolute_delta": float(values.mean()),
            "p95_absolute_delta": float(np.percentile(values, 95)),
            "maximum_absolute_delta": float(values.max()),
        }

    return {
        "sample_count": min(sample_count, len(tensors)),
        "logits": delta_summary(logit_deltas),
        "probabilities": delta_summary(probability_deltas),
        "boxes": delta_summary(box_deltas),
        "query_top_class_agreement": float(np.concatenate(top_class_agreements).mean()),
        "image_top_query_agreement": float(np.concatenate(top_query_agreements).mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import nncf
    import onnx

    tensors = load_detector_calibration_tensors(
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
    presets = {
        "mixed": nncf.QuantizationPreset.MIXED,
        "performance": nncf.QuantizationPreset.PERFORMANCE,
    }
    target_devices = {
        "cpu": nncf.TargetDevice.CPU,
        "gpu": nncf.TargetDevice.GPU,
    }
    smooth_quant_value = -1.0 if args.smooth_quant_alpha is None else args.smooth_quant_alpha
    candidate = nncf.quantize(
        onnx.load(args.source_model),
        calibration_dataset,
        preset=presets[args.preset],
        target_device=target_devices[args.target_device],
        subset_size=len(tensors),
        model_type=nncf.ModelType.TRANSFORMER if args.transformer else None,
        ignored_scope=(
            None if not args.ignore_pattern else nncf.IgnoredScope(patterns=args.ignore_pattern)
        ),
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(
                convolution=smooth_quant_value,
                matmul=smooth_quant_value,
            )
        ),
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, args.output_model)
    try:
        parity: dict[str, Any] = _detector_parity(
            args.source_model,
            args.output_model,
            input_name=args.input_name,
            logits_output=args.logits_output,
            boxes_output=args.boxes_output,
            tensors=tensors,
            sample_count=args.parity_samples,
        )
        parity["execution_provider"] = "cpu"
    except Exception as exc:  # noqa: BLE001 - diagnostic must retain the converted graph
        parity = {
            "sample_count": 0,
            "execution_provider": "cpu",
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    report = {
        "schema_version": "1.0",
        "evaluation": "nncf_onnx_int8_detector_probe",
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
            "sampling": "evenly_spaced_manifest_rows",
            "preset": args.preset,
            "target_device": args.target_device,
            "transformer": bool(args.transformer),
            "smooth_quant_alpha": args.smooth_quant_alpha,
            "ignored_scope_patterns": args.ignore_pattern,
        },
        "detector_output_parity": parity,
        "limitations": [
            "calibration and parity are development diagnostics, not an independent test set",
            "decoded detections and final decisions are evaluated by the full HTTP pipeline",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize a D-FINE ONNX detector with NNCF")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-size", type=int, default=480)
    parser.add_argument("--maximum-samples", type=int, default=128)
    parser.add_argument("--parity-samples", type=int, default=24)
    parser.add_argument("--preset", choices=("mixed", "performance"), default="mixed")
    parser.add_argument("--target-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--smooth-quant-alpha", type=float)
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--mean", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--std", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
