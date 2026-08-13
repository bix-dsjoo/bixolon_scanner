from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ..imaging import decode_image
from ..inference import build_onnx_adapters
from ..package import load_model_package
from ..pipeline import DecisionPipeline
from .rpc_context_rejector import runtime_context_features
from .rpc_data_scale import LEVELS


class _CaptureDetector:
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.version = adapter.version
        self.last_result: Any | None = None

    def detect(self, image: Any) -> Any:
        self.last_result = self.adapter.detect(image)
        return self.last_result


class _CaptureClassifier:
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.version = adapter.version
        self.last_logits: np.ndarray | None = None
        self.call_count = 0

    def classify(self, image: Any, detections: Any) -> np.ndarray:
        self.call_count += 1
        self.last_logits = self.adapter.classify(image, detections)
        return self.last_logits


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _order_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(f"rpc-validation-benchmark:{record['image_id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--context-onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--images-per-level", type=int, default=200)
    args = parser.parse_args()

    package = load_model_package(args.package_dir)
    detector_adapter, classifier_adapter, provider = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    detector = _CaptureDetector(detector_adapter)
    classifier = _CaptureClassifier(classifier_adapter)
    pipeline = DecisionPipeline(
        detector,
        classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
    )
    context = ort.InferenceSession(
        str(args.context_onnx), providers=["CPUExecutionProvider"]
    )
    context.run(
        ["quality_score"], {"features": np.zeros((1, 22), dtype=np.float32)}
    )
    records = [row for row in _read_jsonl(args.manifest) if row["role"] == "selection"]
    selected: dict[str, list[dict[str, Any]]] = {}
    for level in LEVELS:
        candidates = sorted(
            (row for row in records if row["level"] == level), key=_order_key
        )
        selected[level] = candidates[: int(args.images_per_level)]
        if len(selected[level]) < int(args.images_per_level):
            raise ValueError(f"insufficient {level} validation benchmark images")
    encoded = {
        int(row["image_id"]): (args.dataset_root / row["image_path"]).read_bytes()
        for rows in selected.values()
        for row in rows
    }

    def run(record: dict[str, Any]) -> tuple[float, bool]:
        started = time.perf_counter_ns()
        image = decode_image(
            encoded[int(record["image_id"])],
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        calls_before = classifier.call_count
        classifier.last_logits = None
        pipeline.scan(image, request_id=f"rpc-benchmark-{record['image_id']}")
        full_path = classifier.call_count > calls_before
        if full_path:
            if detector.last_result is None or classifier.last_logits is None:
                raise RuntimeError("full Worker path did not capture model outputs")
            features = runtime_context_features(
                detector.last_result.detections,
                classifier.last_logits,
                int(record["width"]),
                int(record["height"]),
                float(package.metadata.classifier.temperature),
            )
            context.run(["quality_score"], {"features": features})
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        return elapsed, full_path

    warmup_records = [row for level in LEVELS for row in selected[level]][
        : int(args.warmup)
    ]
    for record in warmup_records:
        run(record)

    report: dict[str, Any] = {
        "provider": provider,
        "warmup_count": len(warmup_records),
        "images_per_level": int(args.images_per_level),
        "onnxruntime_version": ort.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "difficulty": {},
    }
    for level in LEVELS:
        all_times: list[float] = []
        full_path_times: list[float] = []
        for record in selected[level]:
            elapsed, full_path = run(record)
            all_times.append(elapsed)
            if full_path:
                full_path_times.append(elapsed)
        report["difficulty"][level] = {
            "all_images": _summary(all_times),
            "full_path": _summary(full_path_times) if full_path_times else None,
            "image_recapture_count": len(all_times) - len(full_path_times),
        }
        print(json.dumps({"level": level, **report["difficulty"][level]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
