from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.errors import PackageValidationError
from bixolon_scanner.contracts.runtime_package_v2 import (
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)


def _metadata(root: Path) -> dict:
    (root / "detector.onnx").write_bytes(b"detector")
    (root / "embedder.onnx").write_bytes(b"embedder")
    license_path = root / "licenses" / "APACHE-2.0.txt"
    license_path.parent.mkdir()
    license_path.write_text("Apache License 2.0\n", encoding="utf-8")
    files = ["detector.onnx", "embedder.onnx", "licenses/APACHE-2.0.txt"]
    return {
        "schema_version": "2.0",
        "worker_version": "2.0.0-rc.7",
        "promotion_status": "independent_test_pending",
        "dataset_version": "test-dataset",
        "detector_policy_version": "2.0.0-rc.7",
        "detector_class_count": 2,
        "detector": {
            "filename": "detector.onnx",
            "version": "2.0.0-rc.7",
            "score_threshold": 0.5,
            "nms_iou_threshold": 0.5,
            "max_queries": 100,
        },
        "embedder": {
            "filename": "embedder.onnx",
            "embedder_id": "test-embedder",
            "version": "2.0.0-rc.7",
            "embedding_dimension": 8,
        },
        "metric_projection": {
            "input_dimension": 8,
            "output_dimension": 8,
        },
        "classifier_policy": {
            "version": "2.0.0-rc.7",
            "prototype_weight": 0.5,
            "support_top_k": 3,
            "approval_minimum_similarity": 0.5,
            "approval_minimum_margin": 0.1,
            "ood_maximum_similarity": 0.1,
            "top3_minimum_similarity": 0.2,
            "catalog_conflict_similarity": 0.9,
        },
        "quality": {},
        "checksums": {filename: sha256_file(root / filename) for filename in files},
        "licenses": {"detector": "Apache-2.0", "classifier": "Apache-2.0"},
        "license_files": ["licenses/APACHE-2.0.txt"],
    }


def _write_package(root: Path, payload: dict) -> None:
    (root / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_runtime_package_v2_loads_checked_license_files(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    _write_package(tmp_path, payload)

    package = load_runtime_package_v2(tmp_path)

    assert package.metadata.license_files == ["licenses/APACHE-2.0.txt"]
    assert package.detector_path == tmp_path.resolve() / "detector.onnx"


def test_runtime_package_v2_accepts_attested_production_status(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["promotion_status"] = "production"
    _write_package(tmp_path, payload)

    assert load_runtime_package_v2(tmp_path).metadata.promotion_status == "production"


def test_runtime_package_v2_rejects_unchecked_license_file(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["checksums"].pop("licenses/APACHE-2.0.txt")
    _write_package(tmp_path, payload)

    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(tmp_path)


def test_runtime_package_v2_rejects_tampered_license_file(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    _write_package(tmp_path, payload)
    (tmp_path / "licenses" / "APACHE-2.0.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(tmp_path)


def test_runtime_package_v2_rejects_license_path_escape(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["license_files"] = ["../outside.txt"]

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)
