"""Experimental model regression using production policy and separate feature spaces."""

from __future__ import annotations

import argparse
import copy
import shutil
import time
from pathlib import Path

from ...configuration import load_json_config
from ...contracts import load_runtime_package_v2, load_store_catalog_package
from ...contracts.catalog import sha256_file
from ...evaluation.three_bakery import score_response, summarize
from ...operations.catalog_activation import build_catalog
from ...operations.classifier_consensus_catalog import build_consensus_catalog
from ...operations.three_bakery_runtime import package
from ...runtime.catalog import ResolutionFallbackCatalogClassifier, build_catalog_classifier
from ...runtime.imaging import decode_image
from ...runtime.inference_timing import collect_inference_timings
from ...training.three_bakery_data import read_jsonl, write_json
from ...worker.runtime_factory import build_worker_runtime
from ...worker.settings import WorkerSettings

MANIFESTS = {
    "log": "artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl",
    "original": "artifacts/retraining/three-bakery-revised300/prepared/original_detection.jsonl",
    "final": "artifacts/retraining/recapture-0.2.1/final300-context/final-inputs.jsonl",
}


def assemble_full_student(config: dict, primary: Path, destination: Path):
    """Bind the original detail Catalog explicitly for a standalone Worker experiment."""
    original = Path(config["runtime"])
    payload = load_json_config(original / "metadata.json")
    student = load_json_config(primary / "runtime/metadata.json")
    payload["embedder"] = student["embedder"]
    payload["sources"]["embedder"] = student["sources"]["embedder"]
    payload["sources"]["roi_integrity"] = student["sources"]["roi_integrity"]
    payload["sources"]["detector"].update(
        architecture="dfine_hgnetv2_s_class_agnostic",
        weight_filename=payload["detector"]["filename"],
        weight_sha256=sha256_file(original / payload["detector"]["filename"]),
    )
    payload["licenses"]["detector"] = "Apache-2.0 (D-FINE)"
    payload["licenses"]["classifier"] = "DINOv3 License (detail and independent verifier)"
    payload["licenses"]["primary_classifier"] = "See structural experiment model notices"
    catalog = destination / "catalog"
    if not catalog.exists():
        shutil.copytree(primary / "catalog", catalog)
        shutil.copytree(Path(config["catalog"]), catalog / "detail-catalog")
    payload["classifier_resolution_fallback"].update(
        catalog_directory="detail-catalog",
        catalog_checksums_sha256=sha256_file(catalog / "detail-catalog/checksums.json"),
    )
    files = {name: original / name for name in payload["checksums"]}
    files[payload["embedder"]["filename"]] = primary / "runtime" / payload["embedder"]["filename"]
    package(destination / "runtime", payload, files)


def assemble_primary(config: dict, model: Path, output: Path):
    if (output / "assembly.json").exists():
        report = load_json_config(output / "assembly.json")
        if report["model_sha256"] != sha256_file(model):
            raise ValueError("student assembly belongs to another model")
        load_runtime_package_v2(output / "runtime")
        load_store_catalog_package(output / "catalog")
        return
    runtime_dir = Path(config["runtime"])
    payload = load_json_config(runtime_dir / "metadata.json")
    detail_filename = payload["classifier_resolution_fallback"]["embedder"]["filename"]
    payload["classifier_resolution_fallback"] = None
    filename = payload["embedder"]["filename"]
    model_hash = sha256_file(model)
    payload["embedder"]["embedder_id"] = "structural-student-" + model_hash[:24]
    payload["sources"]["embedder"] = {
        "architecture": model.parent.name,
        "revision": "sha256:" + model_hash,
    }
    payload["sources"]["roi_integrity"] = {
        "architecture": "student_shared_integrity_head",
        "revision": "sha256:" + model_hash,
    }
    files = {name: runtime_dir / name for name in payload["checksums"]}
    files.pop(detail_filename)
    files[filename] = model
    package(output / "runtime", payload, files)
    support = (
        Path(config["source_work"]) / "candidates/dfine-margin-basic-20260908/catalog-support.jsonl"
    )
    for name in ("primary", "rotation"):
        view = copy.deepcopy(payload)
        view["classifier_verification"] = None
        view_files = dict(files)
        view_files.pop(payload["classifier_verification"]["independent_embedder"]["filename"])
        if name == "rotation":
            view["embedder"]["rotation_180_tta"] = True
        package(output / f"runtime-{name}", view, view_files)
        if not (output / f"catalog-{name}").exists():
            build_catalog(
                output / f"runtime-{name}",
                Path(config["source_work"]) / "prepared",
                support,
                output / f"catalog-{name}",
                store_id="three_bakery",
                catalog_version=payload["worker_version"],
                signing_key=None,
                key_id=None,
                authentication="CHECKSUM-SHA256",
                supports_per_class=10,
                provider="cpu",
                cuda_dll_dir=None,
                decision_head="classifier_logits",
            )
    build_consensus_catalog(
        output / "catalog-primary",
        output / "catalog-rotation",
        Path(config["catalog"]) / "independent-verifier",
        output / "catalog",
    )
    write_json(
        output / "assembly.json",
        {
            "model_sha256": model_hash,
            "support_manifest_sha256": sha256_file(support),
            "calibration": "unmodified policy diagnostic; no independent validation or selection",
            "detail_catalog": "original model's original catalog, supplied separately by diagnostic runner",
        },
    )


def evaluate(
    config: dict,
    output: Path,
    dataset: str,
    *,
    primary: Path | None = None,
    detector_runtime: Path | None = None,
    catalog_dir: Path | None = None,
    embedder_provider: str = "same",
    repetitions: int = 1,
    timing_scope: str = "uncontrolled build-host diagnostic",
):
    destination = output / f"{dataset}.json"
    if destination.exists():
        return load_json_config(destination)["summary"]
    runtime = build_worker_runtime(
        WorkerSettings(
            package_dir=detector_runtime or Path(config["runtime"]),
            catalog_dir=catalog_dir or Path(config["catalog"]),
            provider="cpu",
            embedder_provider=embedder_provider,
            embedder_fallback_provider="none",
            verifier_provider="cpu",
            openvino_gpu_precision="f16",
            cpu_detector_intra_op_threads=4,
            cpu_embedder_intra_op_threads=4,
            reuse_verifier_embeddings=True,
        )
    )
    original_classifier = runtime.pipeline.classifier
    new_primary = None
    if primary is not None:
        new_primary, _ = build_catalog_classifier(
            load_runtime_package_v2(primary / "runtime"),
            load_store_catalog_package(primary / "catalog"),
            "cpu",
            cpu_intra_op_threads=4,
            verifier_provider="cpu",
        )
        runtime.pipeline.classifier = ResolutionFallbackCatalogClassifier(
            new_primary,
            original_classifier.fallback_classifier,
            owned_embedders=(),
            resolution_fallback_metadata=original_classifier.resolution_fallback_metadata,
        )
        runtime.pipeline.classifier_metadata = new_primary.metadata
    records = read_jsonl(Path(MANIFESTS[dataset]))
    rows = []
    try:
        for repeat in range(repetitions):
            for index, record in enumerate(records):
                path = Path(record["image_path"])
                if sha256_file(path) != record["image_sha256"]:
                    raise ValueError("regression input changed")
                with decode_image(
                    path.read_bytes(),
                    max_bytes=30000000,
                    max_pixels=50000000,
                    jpeg_draft_size=runtime.jpeg_draft_size,
                ) as image:
                    with collect_inference_timings() as calls:
                        start = time.perf_counter()
                        response = runtime.scan(image, f"structural-{dataset}-{repeat}-{index}")
                        elapsed = (time.perf_counter() - start) * 1000
                rows.append(
                    {
                        "image_id": record["image_id"],
                        "image_sha256": record["image_sha256"],
                        "repetition": repeat,
                        "difficulty": record.get("difficulty", "unspecified"),
                        "elapsed_ms": elapsed,
                        "http_status": 200,
                        "response": response.model_dump(mode="json"),
                        "model_calls": calls,
                        "metrics": score_response(response, record["annotations"], threshold=0.5),
                    }
                )
                if (index + 1) % 33 == 0:
                    print(f"{output.name} {dataset} {index + 1}/{len(records)}", flush=True)
    finally:
        runtime.close()
        if new_primary is not None:
            new_primary.close()
            original_classifier.close()
    summary = summarize(rows)
    write_json(
        destination,
        {
            "scope": timing_scope + "; excludes HTTP/decode; not N100",
            "manifest_sha256": sha256_file(Path(MANIFESTS[dataset])),
            "summary": summary,
            "rows": rows,
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=["primary", "detector"], required=True)
    parser.add_argument(
        "--detector-architecture",
        default="dfine_hgnetv2_n",
        help="Source architecture provenance for a replacement detector",
    )
    parser.add_argument("--datasets", nargs="+", choices=list(MANIFESTS), default=list(MANIFESTS))
    args = parser.parse_args()
    config = load_json_config(args.config)
    root = args.model.parent / "regression"
    if args.kind == "primary":
        assemble_primary(config, args.model, root / "assembly")
        for dataset in args.datasets:
            evaluate(config, root, dataset, primary=root / "assembly")
    else:
        original = Path(config["runtime"])
        payload = load_json_config(original / "metadata.json")
        files = {name: original / name for name in payload["checksums"]}
        files[payload["detector"]["filename"]] = args.model
        payload["sources"]["detector"] = {
            "architecture": args.detector_architecture,
            "revision": "sha256:" + sha256_file(args.model),
        }
        payload["licenses"]["detector"] = "Apache-2.0 (D-FINE)"
        package(root / "runtime", payload, files)
        for dataset in args.datasets:
            evaluate(config, root, dataset, detector_runtime=root / "runtime")


if __name__ == "__main__":
    main()
