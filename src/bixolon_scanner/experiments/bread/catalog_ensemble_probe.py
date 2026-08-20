from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ...contracts.catalog import load_store_catalog_package, sha256_file
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.ports import Detection
from ...runtime.catalog import OnnxCatalogClassifier, OnnxEmbedder
from ...runtime.imaging import decode_image


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _classifier(
    runtime_dir: Path,
    catalog_dir: Path,
    *,
    signing_key: bytes,
    store_id: str,
    key_id: str,
    provider: str,
    cuda_dll_dir: Path | None,
) -> tuple[object, OnnxCatalogClassifier]:
    runtime = load_runtime_package_v2(runtime_dir)
    catalog = load_store_catalog_package(
        catalog_dir,
        signing_key=signing_key,
        expected_store_id=store_id,
        expected_key_id=key_id,
    )
    embedder = OnnxEmbedder(runtime, provider, cuda_dll_dir)
    return runtime, OnnxCatalogClassifier(runtime, catalog, embedder)


def _extract(
    args: argparse.Namespace,
    signing_key: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    runtime_a, classifier_a = _classifier(
        args.runtime_a,
        args.catalog_a,
        signing_key=signing_key,
        store_id=args.store_id,
        key_id=args.key_id,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    runtime_b, classifier_b = _classifier(
        args.runtime_b,
        args.catalog_b,
        signing_key=signing_key,
        store_id=args.store_id,
        key_id=args.key_id,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    class_ids = [label.class_id for label in classifier_a.metadata.labels]
    if class_ids != [label.class_id for label in classifier_b.metadata.labels]:
        raise ValueError("Catalog ensemble label order differs")
    records = {int(row["image_id"]): row for row in _jsonl(args.detector_manifest)}
    logits_a = []
    logits_b = []
    targets = []
    for trace in _jsonl(args.trace):
        if trace["status"] != "SEGMENTATION":
            continue
        segmentations = trace["decision"]["segmentations"]
        detections = [
            Detection(
                float(row["bbox"]["x"]),
                float(row["bbox"]["y"]),
                float(row["bbox"]["x"] + row["bbox"]["width"]),
                float(row["bbox"]["y"] + row["bbox"]["height"]),
                1.0,
            )
            for row in segmentations
        ]
        record = records[int(trace["image_id"])]
        image = decode_image(
            (args.dataset_root / record["image_path"]).read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=runtime_a.metadata.input.jpeg_draft_size,
        )
        try:
            first = classifier_a.classify(image, detections).logits
            second = classifier_b.classify(image, detections).logits
        finally:
            image.close()
        if runtime_a.metadata.input.jpeg_draft_size != runtime_b.metadata.input.jpeg_draft_size:
            raise ValueError("ensemble probe requires an identical JPEG draft policy")
        for row in trace["matched_classifier_diagnostics"]:
            index = int(row["detection_index"])
            logits_a.append(first[index])
            logits_b.append(second[index])
            targets.append(class_ids.index(str(row["target_class_id"])))
    return (
        np.stack(logits_a).astype(np.float32),
        np.stack(logits_b).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        class_ids,
    )


def _normalize_logits(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.mean(axis=1, keepdims=True)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True).clip(min=1e-12)


def _metrics(logits: np.ndarray, targets: np.ndarray, *, allowed_errors: int) -> dict:
    order = np.argsort(-logits, axis=1, kind="stable")
    correct = order[:, 0] == targets
    top3 = np.any(order[:, :3] == targets[:, None], axis=1)
    sorted_logits = np.take_along_axis(logits, order, axis=1)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]) / np.linalg.norm(logits, axis=1).clip(
        min=1e-12
    )
    approved_count = 0
    approved_errors = 0
    threshold = None
    for score in sorted(set(float(value) for value in margin), reverse=True):
        group = margin == score
        next_errors = approved_errors + int(np.count_nonzero(group & ~correct))
        if next_errors > allowed_errors:
            break
        approved_count += int(np.count_nonzero(group))
        approved_errors = next_errors
        threshold = score
    return {
        "sample_count": len(targets),
        "top1_correct_count": int(np.count_nonzero(correct)),
        "top1_accuracy": float(np.mean(correct)),
        "top3_correct_count": int(np.count_nonzero(top3)),
        "top3_accuracy": float(np.mean(top3)),
        "safe_approved_count": approved_count,
        "safe_approved_rate_over_matched": approved_count / len(targets),
        "safe_approved_error_count": approved_errors,
        "approval_threshold": threshold,
    }


def run(args: argparse.Namespace) -> dict:
    signing_key = os.environ.get(args.signing_key_env, "").encode()
    if len(signing_key) < 16:
        raise ValueError("Catalog signing key must contain at least 16 bytes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "logits.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            logits_a = cache["logits_a"].copy()
            logits_b = cache["logits_b"].copy()
            targets = cache["targets"].copy()
            class_ids = cache["class_ids"].tolist()
    else:
        logits_a, logits_b, targets, class_ids = _extract(args, signing_key)
        np.savez_compressed(
            cache_path,
            logits_a=logits_a,
            logits_b=logits_b,
            targets=targets,
            class_ids=np.asarray(class_ids),
        )
    normalized_a = _normalize_logits(logits_a)
    normalized_b = _normalize_logits(logits_b)
    allowed_errors = int(np.floor(args.maximum_error_rate * args.ground_truth_count))
    candidates = []
    for weight_a in np.linspace(0.0, 1.0, 21):
        combined = normalized_a * np.float32(weight_a) + normalized_b * np.float32(1.0 - weight_a)
        candidates.append(
            {
                "weight_a": float(weight_a),
                "weight_b": float(1.0 - weight_a),
                **_metrics(combined, targets, allowed_errors=allowed_errors),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["safe_approved_count"],
            row["top1_correct_count"],
            row["top3_correct_count"],
        ),
    )
    report = {
        "schema_version": "2.0",
        "candidate_id": "frozen-catalog-logit-ensemble",
        "evidence_role": "development_probe",
        "promotion_evidence": False,
        "source": {
            "runtime_a_metadata_sha256": sha256_file(args.runtime_a / "metadata.json"),
            "runtime_b_metadata_sha256": sha256_file(args.runtime_b / "metadata.json"),
            "catalog_a_metadata_sha256": sha256_file(args.catalog_a / "catalog.json"),
            "catalog_b_metadata_sha256": sha256_file(args.catalog_b / "catalog.json"),
            "trace_sha256": sha256_file(args.trace),
            "sample_count": len(targets),
            "class_count": len(class_ids),
        },
        "allowed_approved_error_count": allowed_errors,
        "model_a": _metrics(normalized_a, targets, allowed_errors=allowed_errors),
        "model_b": _metrics(normalized_b, targets, allowed_errors=allowed_errors),
        "selected": selected,
        "candidates": candidates,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"model_a": report["model_a"], "model_b": report["model_b"], "selected": selected},
            indent=2,
        )
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe a two-backbone Store Catalog ensemble")
    parser.add_argument("--runtime-a", type=Path, required=True)
    parser.add_argument("--catalog-a", type=Path, required=True)
    parser.add_argument("--runtime-b", type=Path, required=True)
    parser.add_argument("--catalog-b", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--maximum-error-rate", type=float, default=0.001)
    parser.add_argument("--ground-truth-count", type=int, default=1410)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
