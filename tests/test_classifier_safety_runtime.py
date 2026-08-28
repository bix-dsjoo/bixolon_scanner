from __future__ import annotations

import json
from pathlib import Path

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2
from bixolon_scanner.operations.classifier_safety_runtime import (
    enable_unknown_dual_verifier_recapture,
)


def _write_runtime(root: Path) -> None:
    for filename, content in (
        ("detector.onnx", b"detector"),
        ("embedder.onnx", b"embedder"),
        ("verifier.onnx", b"verifier"),
    ):
        (root / filename).write_bytes(content)
    license_path = root / "licenses/APACHE-2.0.txt"
    license_path.parent.mkdir()
    license_path.write_text("Apache License 2.0\n", encoding="utf-8")
    files = [
        "detector.onnx",
        "embedder.onnx",
        "verifier.onnx",
        "licenses/APACHE-2.0.txt",
    ]
    payload = {
        "schema_version": "2.0",
        "worker_version": "0.1.7",
        "dataset_version": "test-dataset",
        "detector_policy_version": "0.1.7",
        "detector_class_count": 2,
        "detector": {
            "filename": "detector.onnx",
            "version": "0.1.7",
            "score_threshold": 0.5,
            "nms_iou_threshold": 0.5,
            "max_queries": 100,
        },
        "embedder": {
            "filename": "embedder.onnx",
            "embedder_id": "test-embedder",
            "version": "0.1.7",
            "embedding_dimension": 8,
        },
        "metric_projection": {"input_dimension": 8, "output_dimension": 8},
        "classifier_policy": {
            "version": "0.1.7",
            "prototype_weight": 0.5,
            "support_top_k": 3,
            "approval_minimum_similarity": 0.5,
            "approval_minimum_margin": 0.1,
            "ood_maximum_similarity": 0.1,
            "top3_minimum_similarity": 0.2,
            "catalog_conflict_similarity": 0.9,
        },
        "classifier_verification": {
            "ambiguity_maximum_approval_score": 0.5,
            "independent_embedder": {
                "filename": "verifier.onnx",
                "embedder_id": "test-verifier",
                "version": "0.1.7",
                "embedding_dimension": 8,
                "fixed_batch_size": 1,
            },
            "independent_metric_projection": {"input_dimension": 8, "output_dimension": 8},
        },
        "quality": {},
        "checksums": {filename: sha256_file(root / filename) for filename in files},
        "licenses": {"detector": "Apache-2.0", "classifier": "Apache-2.0"},
        "license_files": ["licenses/APACHE-2.0.txt"],
    }
    (root / "metadata.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_enable_unknown_dual_verifier_recapture_preserves_model_payloads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.mkdir()
    _write_runtime(source)

    report = enable_unknown_dual_verifier_recapture(source, candidate)
    loaded = load_runtime_package_v2(candidate)

    assert report["operation"] == "enable_unknown_dual_verifier_recapture"
    assert report["model_graph_or_weight_changed"] is False
    assert report["source_manifest_sha256"] != report["candidate_manifest_sha256"]
    assert (
        loaded.metadata.classifier_verification
        and loaded.metadata.classifier_verification.unknown_recapture_on_dual_verifier_rejection
    )
    for filename in ("detector.onnx", "embedder.onnx", "verifier.onnx"):
        assert sha256_file(candidate / filename) == sha256_file(source / filename)
