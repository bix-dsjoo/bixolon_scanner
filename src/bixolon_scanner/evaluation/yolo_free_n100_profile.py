from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..pipeline.decision import DecisionPipeline
from ..runtime.assisted_detector import attach_classifier_assisted_detector
from ..runtime.catalog import build_catalog_classifier
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image


class _StageHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() != "scan_complete":
            return
        self.rows.append(
            {
                "detector_ms": float(record.detector_ms),
                "classifier_ms": float(record.classifier_ms),
                "processing_time_ms": float(record.processing_time_ms),
            }
        )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_index(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {sha256_file(path): path for path in root.iterdir() if path.is_file()}


def _request_id(image_id: int, iteration: int) -> str:
    return hashlib.sha256(f"n100-profile:{image_id}:{iteration}".encode()).hexdigest()[:32]


def evaluate(args: argparse.Namespace) -> dict:
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(args.catalog, expected_store_id=args.store_id)
    rows = _jsonl(args.manifest)
    image_sha256_index = _sha256_index(args.image_sha_root)
    if args.image_ids:
        requested = set(args.image_ids)
        rows = [row for row in rows if int(row["image_id"]) in requested]
        if {int(row["image_id"]) for row in rows} != requested:
            raise ValueError("one or more requested image IDs were not found")
    embedder_provider = (
        args.provider if args.embedder_provider == "same" else args.embedder_provider
    )
    hybrid_openvino = args.provider == "openvino" and embedder_provider == "openvino_gpu"
    detector = build_detector_v2(
        runtime,
        args.provider,
        args.cuda_dll_dir,
        cpu_detector_workers=args.cpu_detector_workers,
        cpu_intra_op_threads=args.cpu_detector_threads,
        openvino_cache_dir=args.openvino_cache_dir,
        count_verifier_provider=(embedder_provider if hybrid_openvino else None),
        parallel_count_verifier=hybrid_openvino,
    )
    classifier, embedder = build_catalog_classifier(
        runtime,
        catalog,
        embedder_provider,
        args.cuda_dll_dir,
        cpu_intra_op_threads=args.cpu_embedder_threads,
        openvino_cache_dir=args.openvino_cache_dir,
    )
    detector = attach_classifier_assisted_detector(detector, classifier)
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier.metadata,
        runtime.metadata.quality,
        runtime.metadata.count_verifier,
        worker_version=runtime.metadata.worker_version,
        embedder_version=runtime.metadata.embedder.version,
        detector_policy_version=runtime.metadata.detector_policy_version,
        classifier_policy_version=runtime.metadata.classifier_policy.version,
        catalog_version=catalog.metadata.catalog_version,
        assisted_policy=(
            None
            if runtime.metadata.detector.ensemble is None
            else runtime.metadata.detector.ensemble.class_verified_selector
        ),
        detector_primary_classifier_routing=(runtime.metadata.detector_primary_classifier_routing),
    )
    trace = []
    stage_handler = _StageHandler()
    pipeline_logger = logging.getLogger("bixolon_scanner.pipeline.decision")
    previous_level = pipeline_logger.level
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(stage_handler)
    try:
        detector.warmup()
        embedder.warmup()
        warmup = getattr(classifier, "warmup", None)
        if callable(warmup):
            warmup()
        warmup_row = next(row for row in rows if row.get("annotations"))
        warmup_path = args.dataset_root / warmup_row["image_path"]
        if not warmup_path.is_file() and image_sha256_index:
            warmup_path = image_sha256_index.get(warmup_row["image_sha256"], warmup_path)
        for iteration in range(args.warmup_count):
            image_path = warmup_path
            image = decode_image(
                image_path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            try:
                pipeline.scan(
                    image,
                    _request_id(int(warmup_row["image_id"]), -iteration - 1),
                )
            finally:
                image.close()
        for iteration, row in enumerate(rows):
            image_path = args.dataset_root / row["image_path"]
            if not image_path.is_file() and image_sha256_index:
                image_path = image_sha256_index.get(row["image_sha256"], image_path)
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError("N100 profile input checksum mismatch")
            started = time.perf_counter()
            image = decode_image(
                image_path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            try:
                response = pipeline.scan(
                    image,
                    _request_id(int(row["image_id"]), iteration),
                )
            finally:
                image.close()
            trace.append(
                {
                    "image_id": int(row["image_id"]),
                    "image_sha256": row["image_sha256"],
                    "http_status": 200,
                    "client_elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "response": response.model_dump(mode="json"),
                    "stage_latency_ms": stage_handler.rows[-1],
                }
            )
            if len(trace) % 25 == 0 or len(trace) == len(rows):
                print(f"N100 profile: {len(trace)}/{len(rows)}", flush=True)
    finally:
        pipeline_logger.removeHandler(stage_handler)
        pipeline_logger.setLevel(previous_level)
        pipeline.close()
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        "".join(json.dumps(row) + "\n" for row in trace),
        encoding="utf-8",
    )
    return {
        "schema_version": "1.0",
        "evaluation": "yolo_free_n100_provider_profile",
        "provider": (
            args.provider
            if args.provider == embedder_provider
            else f"{args.provider}+{embedder_provider}"
        ),
        "image_count": len(trace),
        "error_count": sum(row["response"]["status"] == "ERROR" for row in trace),
        "trace_sha256": sha256_file(args.trace_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-sha-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda", "openvino"), default="openvino")
    parser.add_argument("--embedder-provider", choices=("same", "openvino_gpu"), default="same")
    parser.add_argument("--openvino-cache-dir", type=Path)
    parser.add_argument("--cpu-detector-workers", type=int, default=1)
    parser.add_argument("--cpu-detector-threads", type=int, default=0)
    parser.add_argument("--cpu-embedder-threads", type=int, default=0)
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument("--image-ids", type=int, nargs="+")
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
