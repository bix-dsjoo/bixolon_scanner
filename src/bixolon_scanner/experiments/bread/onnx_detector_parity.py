from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.onnx import sigmoid
from ...runtime.onnx_session import OrtRunner
from .onnx_int8_probe import _sha256
from .onnx_nncf_detector_int8_probe import load_detector_calibration_tensors


def _delta_summary(parts: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate(parts)
    return {
        "mean_absolute_delta": float(values.mean()),
        "p95_absolute_delta": float(np.percentile(values, 95)),
        "maximum_absolute_delta": float(values.max()),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    tensors = load_detector_calibration_tensors(
        args.manifest,
        args.dataset_root,
        input_size=args.input_size,
        maximum_samples=args.sample_count,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )
    source = OrtRunner(
        args.source_model,
        args.provider,
        cpu_intra_op_threads=args.cpu_threads,
        openvino_cache_dir=args.openvino_cache_dir,
    )
    candidate = OrtRunner(
        args.candidate_model,
        args.provider,
        cpu_intra_op_threads=args.cpu_threads,
        openvino_cache_dir=args.openvino_cache_dir,
    )
    logit_deltas = []
    probability_deltas = []
    box_deltas = []
    top_class_agreements = []
    top_query_agreements = []
    outputs = [args.logits_output, args.boxes_output]
    try:
        for tensor in tensors:
            expected_logits, expected_boxes = source.run(outputs, args.input_name, tensor)
            actual_logits, actual_boxes = candidate.run(outputs, args.input_name, tensor)
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
            probability_deltas.append(
                np.abs(expected_probabilities - actual_probabilities).reshape(-1)
            )
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
                int(np.argmax(expected_scores)) == int(np.argmax(actual_scores))
            )
    finally:
        source.close()
        candidate.close()
    report = {
        "schema_version": "1.0",
        "evaluation": "onnx_detector_output_parity",
        "source_model_sha256": _sha256(args.source_model),
        "candidate_model_sha256": _sha256(args.candidate_model),
        "provider": args.provider,
        "sample_count": len(tensors),
        "logits": _delta_summary(logit_deltas),
        "probabilities": _delta_summary(probability_deltas),
        "boxes": _delta_summary(box_deltas),
        "query_top_class_agreement": float(np.concatenate(top_class_agreements).mean()),
        "image_top_query_agreement": float(np.mean(top_query_agreements)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two ONNX detector outputs")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "openvino"), default="openvino")
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-size", type=int, default=480)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--openvino-cache-dir", type=Path)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--std", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    compare(parser.parse_args(argv))


if __name__ == "__main__":
    main()
