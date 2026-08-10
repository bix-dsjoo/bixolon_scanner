from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .inference import build_onnx_adapters
from .imaging import decode_image
from .package import load_model_package
from .pipeline import DecisionPipeline


class _TimedDetector:
    def __init__(self, adapter):
        self.adapter = adapter
        self.version = adapter.version
        self.last_ms = 0.0

    def detect(self, image):
        started = time.perf_counter()
        result = self.adapter.detect(image)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return result


class _TimedClassifier:
    def __init__(self, adapter):
        self.adapter = adapter
        self.version = adapter.version
        self.last_ms = 0.0

    def classify(self, image, detections):
        started = time.perf_counter()
        result = self.adapter.classify(image, detections)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return result


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _gpu_details() -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        name, driver = completed.stdout.splitlines()[0].split(",", 1)
        return {"name": name.strip(), "driver_version": driver.strip()}
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a packaged ONNX Worker pipeline")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--provider", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = load_model_package(args.package_dir)
    detector, classifier, provider = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    timed_detector = _TimedDetector(detector)
    timed_classifier = _TimedClassifier(classifier)
    pipeline = DecisionPipeline(
        timed_detector,
        timed_classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
    )
    paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise RuntimeError("no benchmark images found")
    encoded_images = []
    for path in paths:
        encoded_images.append(path.read_bytes())
    max_encoded_bytes = max(len(value) for value in encoded_images)
    for index in range(args.warmup):
        image = decode_image(
            encoded_images[index % len(encoded_images)],
            max_bytes=max_encoded_bytes,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        pipeline.scan(image, uuid.uuid4().hex)
    latencies = []
    by_path: dict[str, list[float]] = {"early_exit": [], "full_path": []}
    by_item_count: dict[int, list[float]] = {}
    statuses: dict[str, int] = {}
    stage_latencies: dict[str, list[float]] = {
        "detector": [],
        "classifier": [],
        "decode": [],
        "decision_overhead": [],
    }
    for index in range(args.runs):
        started = time.perf_counter()
        decode_started = time.perf_counter()
        image = decode_image(
            encoded_images[index % len(encoded_images)],
            max_bytes=max_encoded_bytes,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        response = pipeline.scan(image, uuid.uuid4().hex)
        latency = (time.perf_counter() - started) * 1000.0
        latencies.append(latency)
        stage_latencies["detector"].append(timed_detector.last_ms)
        stage_latencies["decode"].append(decode_ms)
        path_kind = "early_exit" if response.model_versions.classifier is None else "full_path"
        by_path[path_kind].append(latency)
        if path_kind == "full_path":
            stage_latencies["classifier"].append(timed_classifier.last_ms)
            stage_latencies["decision_overhead"].append(
                max(
                    0.0,
                    latency
                    - decode_ms
                    - timed_detector.last_ms
                    - timed_classifier.last_ms,
                )
            )
            by_item_count.setdefault(len(response.items), []).append(latency)
        statuses[response.status.value] = statuses.get(response.status.value, 0) + 1
    import onnxruntime as ort

    report = {
        "package_version": package.metadata.package_version,
        "dataset_version": package.metadata.dataset_version,
        "provider": provider,
        "warmup_count": args.warmup,
        "image_files": [path.name for path in paths],
        **_latency_summary(latencies),
        "by_path": {
            name: _latency_summary(values) for name, values in by_path.items() if values
        },
        "full_path_by_item_count": {
            str(count): _latency_summary(values) for count, values in sorted(by_item_count.items())
        },
        "statuses": statuses,
        "stage_latency_ms": {
            name: _latency_summary(values)
            for name, values in stage_latencies.items()
            if values
        },
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "gpu": _gpu_details() if provider == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
