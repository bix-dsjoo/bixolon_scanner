from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bixolon_scanner.operations.release_composition import (
    ReleaseComposition,
    sha256_directory,
    verify_release_composition,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repository(root: Path) -> ReleaseComposition:
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    detector_manifest = manifests / "detector.jsonl"
    classifier_manifest = manifests / "classifier.jsonl"
    detector_manifest.write_text("{}\n", encoding="utf-8")
    classifier_manifest.write_text("{}\n", encoding="utf-8")
    detector_contract_payload = {
        "component": "detector",
        "pipeline_version": "1.0.0",
        "dataset": {
            "dataset_version": "bread-detector-test",
            "manifest_path": "manifests/detector.jsonl",
        },
    }
    classifier_contract_payload = {
        "component": "classifier",
        "pipeline_version": "1.0.0",
        "dataset": {
            "dataset_version": "bread-classifier-test",
            "manifest_path": "manifests/classifier.jsonl",
        },
    }
    contracts = root / "configs" / "training"
    detector_contract = contracts / "detector.json"
    classifier_contract = contracts / "classifier.json"
    _write(detector_contract, detector_contract_payload)
    _write(classifier_contract, classifier_contract_payload)
    package = root / "artifacts" / "packages" / "worker"
    package.mkdir(parents=True)
    detector = package / "detector.onnx"
    classifier = package / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = package / "metadata.json"
    _write(
        metadata,
        {
            "schema_version": "2.1",
            "worker_version": "1.0.0",
            "promotion_status": "production",
            "dataset_version": "bread-test",
            "detector": {
                "filename": "detector.onnx",
                "version": "1.0.0",
                "score_threshold": 0.5,
                "nms_iou_threshold": 0.7,
                "max_queries": 300,
            },
            "classifier": {
                "filename": "classifier.onnx",
                "version": "1.0.0",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "approval_threshold": 0.9,
                "temperature": 1.0,
                "labels": [{"class_id": "bread_01", "class_name": "Bread"}],
            },
            "quality": {},
            "checksums": {"detector.onnx": _sha(detector), "classifier.onnx": _sha(classifier)},
            "licenses": {"detector": "test", "classifier": "test"},
            "sources": {
                "detector": {
                    "architecture": "test",
                    "revision": "a" * 40,
                    "training_pipeline_version": "1.0.0",
                    "training_contract_sha256": _canonical_sha(detector_contract_payload),
                    "training_dataset_version": "bread-detector-test",
                    "training_manifest_sha256": _sha(detector_manifest),
                },
                "classifier": {
                    "architecture": "test",
                    "revision": "b" * 40,
                    "training_pipeline_version": "1.0.0",
                    "training_contract_sha256": _canonical_sha(classifier_contract_payload),
                    "training_dataset_version": "bread-classifier-test",
                    "training_manifest_sha256": _sha(classifier_manifest),
                },
            },
            "promotion": {
                "decision": "approved",
                "method": "all_gates",
                "decided_on": "2026-08-14",
            },
        },
    )
    pubspec = root / "apps" / "product_scanner" / "pubspec.yaml"
    pubspec.parent.mkdir(parents=True)
    pubspec.write_text("name: product_scanner\nversion: 1.0.0+2\n", encoding="utf-8")
    return ReleaseComposition.model_validate(
        {
            "schema_version": "1.0",
            "release": "test-release",
            "lifecycle": "locked",
            "versions": {
                "python": "1.0.0",
                "worker": "1.0.0",
                "detector": "1.0.0",
                "classifier": "1.0.0",
                "detector_training_pipeline": "1.0.0",
                "classifier_training_pipeline": "1.0.0",
                "dataset": "bread-test",
                "app": "1.0.0+2",
            },
            "model_package": {
                "path": "artifacts/packages/worker",
                "sha256": sha256_directory(package),
            },
            "model_package_metadata": {
                "path": "artifacts/packages/worker/metadata.json",
                "sha256": _sha(metadata),
            },
            "training_contracts": {
                "detector": {
                    "path": "configs/training/detector.json",
                    "sha256": _sha(detector_contract),
                },
                "classifier": {
                    "path": "configs/training/classifier.json",
                    "sha256": _sha(classifier_contract),
                },
            },
            "independent_test_status": "pending_user_images",
        }
    )


def test_release_composition_locks_all_independent_versions(tmp_path: Path) -> None:
    result = verify_release_composition(_repository(tmp_path), repository_root=tmp_path)
    assert result["passed"] is True
    assert result["bundle_locked"] is False


def test_release_composition_rejects_app_version_drift(tmp_path: Path) -> None:
    composition = _repository(tmp_path)
    (tmp_path / "apps" / "product_scanner" / "pubspec.yaml").write_text(
        "name: product_scanner\nversion: 1.0.1+3\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Flutter app version"):
        verify_release_composition(composition, repository_root=tmp_path)


def test_release_composition_verifies_every_windows_bundle_file(tmp_path: Path) -> None:
    composition = _repository(tmp_path)
    bundle = tmp_path / "artifacts" / "releases" / "app"
    package = tmp_path / "artifacts" / "packages" / "worker"
    files = {
        "product_scanner.exe": b"app",
        "worker/bixolon-worker.exe": b"worker",
        "worker/model-package/metadata.json": (package / "metadata.json").read_bytes(),
        "worker/model-package/detector.onnx": (package / "detector.onnx").read_bytes(),
        "worker/model-package/classifier.onnx": (package / "classifier.onnx").read_bytes(),
        "worker/cuda-runtime/cudart64_13.dll": b"cuda",
        "worker/cuda-runtime/cublas64_13.dll": b"cublas",
        "worker/cuda-runtime/cudnn64_9.dll": b"cudnn",
        "worker/cuda-runtime/nvJitLink_130_0.dll": b"nvjitlink",
        "worker/cuda-runtime/nvrtc64_130_0.dll": b"nvrtc",
        "worker/cuda-runtime/nvrtc-builtins64_130.dll": b"nvrtc-builtins",
    }
    for relative, content in files.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    rows = [
        {"path": relative, "size_bytes": len(content), "sha256": _sha(bundle / relative)}
        for relative, content in sorted(files.items())
    ]
    manifest = bundle / "bundle-manifest.json"
    _write(
        manifest,
        {
            "schema_version": "1.0",
            "release": composition.release,
            "app_version": composition.versions.app,
            "file_count": len(rows),
            "files": rows,
        },
    )
    composition_payload = composition.model_dump(mode="json")
    composition_payload["windows_bundle_manifest"] = {
        "path": "artifacts/releases/app/bundle-manifest.json",
        "sha256": _sha(manifest),
    }
    composition = ReleaseComposition.model_validate(composition_payload)

    assert verify_release_composition(composition, repository_root=tmp_path, require_bundle=True)[
        "bundle"
    ]["file_count"] == len(rows)
    (bundle / "product_scanner.exe").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release_composition(composition, repository_root=tmp_path, require_bundle=True)
