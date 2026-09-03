from __future__ import annotations

import json
from pathlib import Path

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.operations.consistent_evidence_runtime import (
    build_consistent_evidence_runtime,
)


def _write_source_runtime(root: Path) -> None:
    files = {
        "detector.onnx": b"detector",
        "embedder.onnx": b"primary-192",
        "embedder-fallback.onnx": b"detail-224",
        "classifier-verifier.onnx": b"verifier-160",
        "licenses/APACHE-2.0.txt": b"Apache License 2.0\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    embedder = {
        "filename": "embedder.onnx",
        "embedder_id": "dinov3-convnext-tiny",
        "version": "0.1.12",
        "input_size": [192, 192],
        "embedding_dimension": 8,
    }
    payload = {
        "schema_version": "2.0",
        "worker_version": "0.1.12",
        "dataset_version": "test-dataset",
        "detector_policy_version": "0.1.12",
        "detector_class_count": 1,
        "detector_class_mode": "class_agnostic",
        "detector": {
            "filename": "detector.onnx",
            "version": "0.1.12",
            "score_threshold": 0.5,
            "nms_iou_threshold": 0.4,
            "max_queries": 100,
        },
        "embedder": embedder,
        "metric_projection": {"input_dimension": 8, "output_dimension": 8},
        "classifier_policy": {
            "version": "0.1.12",
            "prototype_weight": 0.5,
            "support_top_k": 3,
            "approval_minimum_similarity": 0.5,
            "approval_minimum_margin": 0.1,
            "ood_maximum_similarity": 0.1,
            "top3_minimum_similarity": 0.2,
            "catalog_conflict_similarity": 0.9,
        },
        "classifier_resolution_fallback": {
            "embedder": {
                **embedder,
                "filename": "embedder-fallback.onnx",
                "input_size": [224, 224],
            },
            "fallback_on_unknown": True,
            "selective_roi_only": True,
        },
        "classifier_verification": {
            "ambiguity_maximum_approval_score": 0.55,
            "independent_embedder": {
                "filename": "classifier-verifier.onnx",
                "embedder_id": "dinov3-vitb16-frozen",
                "version": "0.1.12",
                "input_size": [160, 160],
                "embedding_dimension": 8,
                "fixed_batch_size": 1,
            },
            "independent_metric_projection": {
                "input_dimension": 8,
                "output_dimension": 8,
            },
        },
        "quality": {},
        "checksums": {name: sha256_file(root / name) for name in files},
        "licenses": {"detector": "Apache-2.0", "classifier": "Apache-2.0"},
        "license_files": ["licenses/APACHE-2.0.txt"],
    }
    (root / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_builder_enables_global_approval_verification_without_changing_payloads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_source_runtime(source)
    output = tmp_path / "candidate"

    report = build_consistent_evidence_runtime(source, output)

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    verification = metadata["classifier_verification"]
    assert verification["verify_all_approved_candidates"] is False
    assert verification["unknown_recapture_on_any_verifier_rejection"] is False
    assert verification["unknown_recapture_on_dual_verifier_rejection"] is False
    assert metadata["embedder"]["input_size"] == [192, 192]
    assert metadata["classifier_resolution_fallback"]["embedder"]["input_size"] == [224, 224]
    assert metadata["classifier_verification"]["independent_embedder"]["input_size"] == [160, 160]
    assert report["model_graph_or_weight_changed"] is False
    assert report["source_manifest_sha256"] != report["candidate_manifest_sha256"]
    for filename in (
        "detector.onnx",
        "embedder.onnx",
        "embedder-fallback.onnx",
        "classifier-verifier.onnx",
    ):
        assert sha256_file(source / filename) == sha256_file(output / filename)
