from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from ..contracts.model_package import load_model_package, sha256_file
from ..pipeline import DecisionPipeline
from ..runtime.imaging import decode_image
from ..runtime.onnx import build_onnx_adapters


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


def _resolve_gpu_selection(
    rows: list[dict[str, str | float | int]], visible: str | None
) -> tuple[dict[str, str | float | int], str]:
    if not rows:
        raise ValueError("nvidia-smi returned no physical GPUs")
    if len(rows) == 1:
        return rows[0], "single_physical_gpu"
    if visible:
        token = visible.split(",", 1)[0].strip()
        if token.casefold().startswith("gpu-"):
            matches = [
                row for row in rows if str(row["uuid"]).casefold().startswith(token.casefold())
            ]
            if len(matches) == 1:
                return matches[0], "CUDA_VISIBLE_DEVICES_UUID"
        source = "ambiguous_CUDA_VISIBLE_DEVICES"
    else:
        source = "ambiguous_default_cuda_device_0"
    return {
        "physical_index": -1,
        "uuid": "",
        "name": "",
        "driver_version": "",
        "memory_total_mib": 0.0,
    }, source


def _gpu_details() -> dict[str, str | float | int] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = []
        for line in completed.stdout.splitlines():
            index, uuid_value, name, driver, memory_mib = line.split(",", 4)
            rows.append(
                {
                    "physical_index": int(index.strip()),
                    "uuid": uuid_value.strip(),
                    "name": name.strip(),
                    "driver_version": driver.strip(),
                    "memory_total_mib": float(memory_mib.strip()),
                }
            )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selected, selection_source = _resolve_gpu_selection(rows, visible)
        details: dict[str, str | float | int] = {
            **selected,
            "physical_gpu_count": float(len(rows)),
            "ort_cuda_device_id": 0,
            "selection_source": selection_source,
            "cuda_visible_devices": visible or "",
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
        }
        version = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        match = re.search(r"CUDA Version:\s*([0-9.]+)", version.stdout)
        if match:
            details["cuda_version"] = match.group(1)
        return details
    except Exception:
        return None


def _system_details() -> dict[str, str | float | None]:
    total_memory_gib: float | None = None
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    windows_build: float | None = None
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total_memory_gib = float(status.total_physical / (1024**3))
        except Exception:
            total_memory_gib = None
        try:
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed.stdout.strip():
                cpu = completed.stdout.strip()
        except Exception:
            pass
        try:
            windows_build = float(platform.version().split(".")[-1])
        except (TypeError, ValueError):
            windows_build = None
    return {
        "operating_system": platform.platform(),
        "windows_build": windows_build,
        "cpu": cpu,
        "memory_total_gib": total_memory_gib,
    }


def _manifest_evidence(
    images_dir: Path,
    image_paths: list[Path],
    manifest_path: Path | None,
    checksums_path: Path | None,
) -> dict[str, object]:
    image_hashes = {path.name: sha256_file(path) for path in image_paths}
    if manifest_path is None:
        if checksums_path is not None:
            raise ValueError("--manifest-checksums requires --image-manifest")
        return {
            "benchmark_manifest_sha256": None,
            "benchmark_manifest_checksums_sha256": None,
            "image_artifact_sha256": image_hashes,
        }
    manifest_path = manifest_path.resolve()
    checksums_path = (checksums_path or manifest_path.parent / "checksums.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("benchmark manifest has no records")
    expected_paths: list[Path] = []
    expected_hashes: dict[str, str] = {}
    root = manifest_path.parent.resolve()
    if images_dir.resolve() != (root / "images").resolve():
        raise ValueError("benchmark --images must be the manifest images directory")
    for record in records:
        relative = str(record["image_path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("benchmark manifest image escapes its root") from exc
        expected_paths.append(path)
        expected_hashes[relative.replace("\\", "/")] = str(record["source_image_sha256"])
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("benchmark manifest contains duplicate image paths")
    if set(expected_paths) != set(image_paths):
        raise ValueError("benchmark image directory does not match the manifest")
    manifest_image_hashes = {
        path.relative_to(root).as_posix(): image_hashes[path.name] for path in image_paths
    }
    if manifest_image_hashes != expected_hashes:
        raise ValueError("benchmark image checksum does not match the manifest")
    ledger = json.loads(checksums_path.read_text(encoding="utf-8"))
    if ledger.get("phase") != "benchmark-manifest" or not isinstance(ledger.get("outputs"), dict):
        raise ValueError("benchmark manifest checksum ledger is invalid")
    for relative, expected in ledger["outputs"].items():
        artifact = (root / str(relative)).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError("benchmark ledger artifact escapes its root") from exc
        relative_key = str(relative).replace("\\", "/")
        actual = manifest_image_hashes.get(relative_key)
        if actual is None and artifact.is_file():
            actual = sha256_file(artifact)
        if not artifact.is_file() or actual != expected:
            raise ValueError(f"benchmark ledger checksum mismatch: {relative}")
    if ledger["outputs"].get(manifest_path.name) != sha256_file(manifest_path):
        raise ValueError("benchmark manifest is not bound by its checksum ledger")
    return {
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "benchmark_manifest_checksums_sha256": sha256_file(checksums_path),
        "image_artifact_sha256": manifest_image_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a packaged ONNX Worker pipeline")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--manifest-checksums", type=Path)
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
    paths = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise RuntimeError("no benchmark images found")
    manifest_evidence = _manifest_evidence(
        args.images.resolve(),
        [path.resolve() for path in paths],
        args.image_manifest,
        args.manifest_checksums,
    )
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
                    latency - decode_ms - timed_detector.last_ms - timed_classifier.last_ms,
                )
            )
            by_item_count.setdefault(len(response.items), []).append(latency)
        statuses[response.status.value] = statuses.get(response.status.value, 0) + 1
    import onnxruntime as ort

    package_artifacts = {
        "metadata.json": sha256_file(args.package_dir / "metadata.json"),
        package.metadata.detector.filename: sha256_file(package.detector_path),
        package.metadata.classifier.filename: sha256_file(package.classifier_path),
    }
    if package.count_verifier_path is not None:
        package_artifacts[package.metadata.count_verifier.filename] = sha256_file(
            package.count_verifier_path
        )

    report = {
        "package_version": package.metadata.package_version,
        "dataset_version": package.metadata.dataset_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "package_artifact_sha256": package_artifacts,
        **manifest_evidence,
        "provider": provider,
        "warmup_count": args.warmup,
        "image_files": [path.name for path in paths],
        **_latency_summary(latencies),
        "by_path": {name: _latency_summary(values) for name, values in by_path.items() if values},
        "full_path_by_item_count": {
            str(count): _latency_summary(values) for count, values in sorted(by_item_count.items())
        },
        "statuses": statuses,
        "stage_latency_ms": {
            name: _latency_summary(values) for name, values in stage_latencies.items() if values
        },
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "onnxruntime_build_info": ort.get_build_info(),
        "system": _system_details(),
        "gpu": _gpu_details() if provider == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
