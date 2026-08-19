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


def test_release_composition_verifies_manual_waiver_provenance(tmp_path: Path) -> None:
    composition = _repository(tmp_path)
    package = tmp_path / "artifacts" / "packages" / "worker"
    metadata_path = package / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    waiver_row = {
        "gate": "evaluation_set_independence",
        "observed": 0.0,
        "target": 1.0,
        "sample_count": 10,
        "correct_count": 0,
        "reason": "Owner accepted a bridge release pending an independent test.",
    }
    metadata["promotion"] = {
        "decision": "approved",
        "method": "manual_waiver",
        "decided_on": "2026-08-19",
        "waivers": [waiver_row],
    }
    _write(metadata_path, metadata)
    waiver_path = tmp_path / "configs" / "releases" / "owner-waiver.json"
    _write(
        waiver_path,
        {
            "release": composition.release,
            "promotion_method": "owner_approved_known_limitations",
            "dataset_version": "bread-test",
            "production_metadata_sha256": _sha(metadata_path),
            "production_package": {
                "path": "artifacts/packages/worker",
                "sha256_directory": sha256_directory(package),
            },
            "versions": {
                "worker_version": "1.0.0",
                "detector_version": "1.0.0",
                "classifier_version": "1.0.0",
            },
            "checks": {"evaluation_set_independence": False},
            "failures": ["evaluation_set_independence"],
            "waivers": [waiver_row],
        },
    )
    payload = composition.model_dump(mode="json")
    payload["schema_version"] = "1.1"
    payload["versions"]["detector_training_pipeline"] = None
    payload["versions"]["classifier_training_pipeline"] = None
    payload["model_package"]["sha256"] = sha256_directory(package)
    payload["model_package_metadata"]["sha256"] = _sha(metadata_path)
    payload["training_contracts"] = None
    payload["manual_waiver"] = {
        "path": "configs/releases/owner-waiver.json",
        "sha256": _sha(waiver_path),
    }
    composition = ReleaseComposition.model_validate(payload)

    result = verify_release_composition(composition, repository_root=tmp_path)

    assert result["provenance"]["mode"] == "manual_waiver"
    assert result["provenance"]["waived_gates"] == ["evaluation_set_independence"]


def test_release_composition_rejects_partial_pipeline_versions(tmp_path: Path) -> None:
    payload = _repository(tmp_path).model_dump(mode="json")
    payload["schema_version"] = "1.1"
    payload["versions"]["detector_training_pipeline"] = None
    payload["training_contracts"] = None
    payload["manual_waiver"] = {
        "path": "configs/releases/owner-waiver.json",
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="pipeline versions"):
        ReleaseComposition.model_validate(payload)


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
