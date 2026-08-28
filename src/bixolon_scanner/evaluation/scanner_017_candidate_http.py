from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..pipeline.decision import DecisionPipeline
from ..runtime.catalog import build_catalog_classifier
from ..runtime.detector_v2 import build_detector_v2
from ..worker.api import create_app
from ..worker.settings import WorkerSettings


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _target_manifest(coco_path: Path, dataset_root: Path) -> list[dict]:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    annotations: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    rows = []
    for image in coco["images"]:
        image_id = int(image["id"])
        path = Path("images") / Path(image["file_name"]).name
        rows.append(
            {
                "image_id": image_id,
                "image_path": path.as_posix(),
                "image_sha256": sha256_file(dataset_root / path),
                "annotations": annotations.get(image_id, []),
            }
        )
    return rows


def _candidate_runtime(runtime, *, enable_unknown_dual_recapture: bool):
    verification = runtime.metadata.classifier_verification
    if verification is None:
        raise ValueError("candidate evaluation requires classifier verification metadata")
    candidate_verification = verification.model_copy(
        update={"unknown_recapture_on_dual_verifier_rejection": enable_unknown_dual_recapture}
    )
    metadata = runtime.metadata.model_copy(
        update={"classifier_verification": candidate_verification}
    )
    return replace(runtime, metadata=metadata)


def evaluate(args: argparse.Namespace) -> dict:
    if (args.manifest is None) == (args.target_coco is None):
        raise ValueError("provide exactly one of --manifest or --target-coco")
    rows = (
        _jsonl(args.manifest)
        if args.manifest is not None
        else _target_manifest(args.target_coco, args.dataset_root)
    )
    if args.normalized_manifest_output is not None:
        args.normalized_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.normalized_manifest_output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    runtime = _candidate_runtime(
        load_runtime_package_v2(args.runtime),
        enable_unknown_dual_recapture=args.enable_unknown_dual_recapture,
    )
    catalog = load_store_catalog_package(args.catalog, expected_store_id=args.store_id)
    detector = build_detector_v2(runtime, args.provider, args.cuda_dll_dir)
    classifier, embedder = build_catalog_classifier(
        runtime,
        catalog,
        args.provider,
        args.cuda_dll_dir,
    )
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
    )
    trace = []
    try:
        detector.warmup()
        embedder.warmup()
        warmup = getattr(classifier, "warmup", None)
        if callable(warmup):
            warmup()
        settings = WorkerSettings(jpeg_draft_size=runtime.metadata.input.jpeg_draft_size)
        app = create_app(settings=settings, pipeline=pipeline)
        warmup_record = next(row for row in rows if row.get("annotations"))
        warmup_path = args.dataset_root / warmup_record["image_path"]
        with TestClient(app) as client:
            for _ in range(args.warmup_count):
                response = client.post(
                    "/v1/scan",
                    files={"image": (warmup_path.name, warmup_path.read_bytes())},
                )
                if response.status_code != 200:
                    raise RuntimeError("candidate HTTP warmup failed")
            for index, row in enumerate(rows, start=1):
                image_path = args.dataset_root / row["image_path"]
                if sha256_file(image_path) != row["image_sha256"]:
                    raise ValueError("candidate HTTP input checksum mismatch")
                started = time.perf_counter()
                response = client.post(
                    "/v1/scan",
                    files={"image": (image_path.name, image_path.read_bytes())},
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                trace.append(
                    {
                        "image_id": int(row["image_id"]),
                        "image_sha256": row["image_sha256"],
                        "http_status": response.status_code,
                        "client_elapsed_ms": elapsed_ms,
                        "response": response.json(),
                    }
                )
                if index % 50 == 0 or index == len(rows):
                    print(f"candidate HTTP: {index}/{len(rows)}", flush=True)
    finally:
        pipeline.close()

    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        "".join(json.dumps(row) + "\n" for row in trace),
        encoding="utf-8",
    )
    return {
        "schema_version": "1.0",
        "product_version": runtime.metadata.worker_version,
        "provider": args.provider,
        "http_transport": "FastAPI TestClient",
        "image_count": len(trace),
        "error_count": sum(
            row["http_status"] != 200 or row["response"].get("status") == "ERROR" for row in trace
        ),
        "unknown_dual_verifier_recapture_enabled": args.enable_unknown_dual_recapture,
        "trace_sha256": sha256_file(args.trace_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--target-coco", type=Path)
    parser.add_argument("--normalized-manifest-output", type=Path)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=5)
    parser.add_argument("--enable-unknown-dual-recapture", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
