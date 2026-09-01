from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...runtime.onnx import prepare_rgb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration_tensors(
    manifest: Path,
    dataset_root: Path,
    *,
    input_size: int,
    maximum_samples: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> list[np.ndarray]:
    if input_size < 32:
        raise ValueError("INT8 calibration input size must be at least 32")
    if maximum_samples < 1:
        raise ValueError("INT8 calibration sample count must be positive")
    root = dataset_root.resolve()
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("INT8 calibration manifest is empty")
    tensors: list[np.ndarray] = []
    for row in rows[:maximum_samples]:
        image_path = (root / str(row["image_path"])).resolve()
        if root not in image_path.parents:
            raise ValueError("INT8 calibration image resolves outside the dataset root")
        if not image_path.is_file():
            raise FileNotFoundError(f"INT8 calibration image is missing: {image_path}")
        expected_sha256 = row.get("image_sha256")
        if expected_sha256 and _sha256(image_path) != expected_sha256:
            raise ValueError("INT8 calibration image checksum mismatch")
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


class CalibrationReader:
    def __init__(self, input_name: str, tensors: list[np.ndarray]):
        self.input_name = input_name
        self.tensors = tensors
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self.tensors):
            return None
        tensor = self.tensors[self._index]
        self._index += 1
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self._index = 0


def _embedding_parity(
    source_model: Path,
    candidate_model: Path,
    *,
    input_name: str,
    output_name: str,
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
    cosine_similarities = []
    maximum_absolute_delta = 0.0
    for tensor in tensors[:sample_count]:
        expected = np.asarray(source.run([output_name], {input_name: tensor})[0], dtype=np.float64)
        actual = np.asarray(candidate.run([output_name], {input_name: tensor})[0], dtype=np.float64)
        maximum_absolute_delta = max(
            maximum_absolute_delta, float(np.max(np.abs(expected - actual)))
        )
        denominator = np.maximum(
            np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1), 1e-12
        )
        cosine_similarities.extend(
            (np.sum(expected * actual, axis=1) / denominator).astype(float).tolist()
        )
    return {
        "sample_count": min(sample_count, len(tensors)),
        "minimum_cosine_similarity": float(np.min(cosine_similarities)),
        "mean_cosine_similarity": float(np.mean(cosine_similarities)),
        "maximum_absolute_delta": maximum_absolute_delta,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    tensors = load_calibration_tensors(
        args.manifest,
        args.dataset_root,
        input_size=args.input_size,
        maximum_samples=args.maximum_samples,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    calibration_methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    activation_types = {
        "int8": QuantType.QInt8,
        "uint8": QuantType.QUInt8,
    }
    quantize_static(
        model_input=str(args.source_model),
        model_output=str(args.output_model),
        calibration_data_reader=CalibrationReader(args.input_name, tensors),
        quant_format=QuantFormat.QDQ,
        activation_type=activation_types[args.activation_type],
        weight_type=QuantType.QInt8,
        calibrate_method=calibration_methods[args.calibration_method],
        per_channel=args.per_channel,
        op_types_to_quantize=args.op_types,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "onnx_static_int8_embedding_probe",
        "source_model": str(args.source_model.resolve()),
        "source_model_sha256": _sha256(args.source_model),
        "source_size_bytes": args.source_model.stat().st_size,
        "candidate_model": str(args.output_model.resolve()),
        "candidate_model_sha256": _sha256(args.output_model),
        "candidate_size_bytes": args.output_model.stat().st_size,
        "calibration": {
            "manifest_sha256": _sha256(args.manifest),
            "sample_count": len(tensors),
            "method": args.calibration_method,
            "per_channel": bool(args.per_channel),
            "activation_type": args.activation_type,
            "op_types": args.op_types,
            "input_size": args.input_size,
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
    parser = argparse.ArgumentParser(
        description="Quantize and measure an ONNX embedder with INT8 PTQ"
    )
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
    parser.add_argument(
        "--calibration-method", choices=("minmax", "entropy", "percentile"), default="minmax"
    )
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--activation-type", choices=("int8", "uint8"), default="int8")
    parser.add_argument("--op-types", nargs="+", default=("Conv", "MatMul", "Gemm"))
    parser.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    parser.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
