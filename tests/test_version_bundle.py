from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import (
    CatalogActivation,
    CatalogLabel,
    CatalogMetadata,
    sha256_file,
)
from bixolon_scanner.operations.version_bundle import (
    VersionBundleConfig,
    prepare_version_bundle,
    verify_prepared_version,
    write_final_bundle_manifest,
)


def _write_runtime(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "detector.onnx").write_bytes(b"detector-graph")
    (root / "embedder.onnx").write_bytes(b"embedder-graph")
    licenses = {
        "licenses/APACHE-2.0.txt": "Apache License 2.0\n",
        "licenses/DINOV3-LICENSE.md": "DINOv3 license\n",
        "licenses/THIRD_PARTY_MODELS.md": "Third-party models\n",
    }
    for relative, content in licenses.items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(content, encoding="utf-8")
    files = ["detector.onnx", "embedder.onnx", *licenses]
    metadata = {
        "schema_version": "2.0",
        "worker_version": "2.0.1-rc.3",
        "promotion_status": "development",
        "dataset_version": "test-dataset",
        "detector_policy_version": "2.0.1-rc.3",
        "detector_class_count": 2,
        "detector": {
            "filename": "detector.onnx",
            "version": "2.0.1-rc.3",
            "score_threshold": 0.5,
            "nms_iou_threshold": 0.5,
            "max_queries": 100,
        },
        "embedder": {
            "filename": "embedder.onnx",
            "embedder_id": "test-embedder",
            "version": "2.0.1-rc.3",
            "embedding_dimension": 8,
        },
        "metric_projection": {"input_dimension": 8, "output_dimension": 8},
        "classifier_policy": {
            "version": "2.0.1-rc.3",
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
        "license_files": list(licenses),
    }
    (root / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")


def _write_catalog(root: Path) -> None:
    root.mkdir(parents=True)
    source_manifest = root / "source-manifest.jsonl"
    source_manifest.write_text('{"image_id":"image-01"}\n', encoding="utf-8")
    metadata = CatalogMetadata(
        catalog_version="2.0.1-rc.3",
        store_id="test-store",
        embedder_id="test-embedder",
        embedder_version="2.0.1-rc.3",
        classifier_policy_version="2.0.1-rc.3",
        embedding_dimension=8,
        support_count_per_class=10,
        support_count=10,
        labels=[
            CatalogLabel(
                class_id="bread_01",
                class_name="Bread",
                support_offset=0,
                support_count=10,
                compactness=0.9,
            )
        ],
        source_manifest_sha256=sha256_file(source_manifest),
    )
    (root / "catalog.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    (root / "activation.json").write_text(
        CatalogActivation(state="active").model_dump_json(), encoding="utf-8"
    )
    (root / "supports.bin").write_bytes(b"supports")
    (root / "prototypes.bin").write_bytes(b"prototypes")
    (root / "statistics.json").write_text("{}", encoding="utf-8")
    checksums = {
        name: sha256_file(root / name)
        for name in (
            "activation.json",
            "catalog.json",
            "prototypes.bin",
            "source-manifest.jsonl",
            "statistics.json",
            "supports.bin",
        )
    }
    (root / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    (root / "signature.json").write_text("{}", encoding="utf-8")


def _config(root: Path) -> VersionBundleConfig:
    runtime = root / "artifacts" / "runtime"
    catalog = root / "artifacts" / "catalog"
    cuda = root / "artifacts" / "cuda"
    evidence = root / "artifacts" / "evaluation.json"
    _write_runtime(runtime)
    _write_catalog(catalog)
    cuda.mkdir()
    (cuda / "cudart64_13.dll").write_bytes(b"cuda")
    (cuda / "cublas64_13.dll").write_bytes(b"cublas")
    (cuda / "cudnn64_9.dll").write_bytes(b"cudnn")
    evidence.write_text("{}", encoding="utf-8")
    return VersionBundleConfig.model_validate(
        {
            "schema_version": "1.0",
            "version": "0.0.1",
            "app_build": 1,
            "source_candidate": "2.0.1-rc.3",
            "runtime": {
                "path": "artifacts/runtime",
                "manifest_sha256": directory_content_manifest(runtime)["manifest_sha256"],
            },
            "catalog": {
                "path": "artifacts/catalog",
                "manifest_sha256": directory_content_manifest(catalog)["manifest_sha256"],
                "store_id": "test-store",
            },
            "cuda_runtime": {
                "path": "artifacts/cuda",
                "manifest_sha256": directory_content_manifest(cuda)["manifest_sha256"],
                "files": ["cublas64_13.dll", "cudart64_13.dll", "cudnn64_9.dll"],
            },
            "evaluation_evidence": [
                {"path": "artifacts/evaluation.json", "sha256": sha256_file(evidence)}
            ],
            "output_root": "artifacts/versions",
        }
    )


def test_version_bundle_relabels_only_metadata_and_keeps_payloads(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = prepare_version_bundle(config, repository_root=tmp_path)

    assert result["passed"] is True
    staging = tmp_path / "artifacts" / "versions" / "0.0.1" / "staging"
    runtime = json.loads((staging / "runtime" / "metadata.json").read_text(encoding="utf-8"))
    catalog = json.loads((staging / "catalog" / "catalog.json").read_text(encoding="utf-8"))
    assert "promotion_status" not in runtime
    assert "promotion" not in runtime
    assert runtime["worker_version"] == "0.0.1"
    assert runtime["detector"]["version"] == "0.0.1"
    assert runtime["embedder"]["version"] == "0.0.1"
    assert runtime["classifier_policy"]["version"] == "0.0.1"
    assert catalog["catalog_version"] == "0.0.1"
    assert catalog["authentication"] == "CHECKSUM-SHA256"
    assert not (staging / "catalog" / "signature.json").exists()
    assert (staging / "runtime" / "detector.onnx").read_bytes() == b"detector-graph"
    assert (staging / "runtime" / "embedder.onnx").read_bytes() == b"embedder-graph"
    assert (staging / "catalog" / "supports.bin").read_bytes() == b"supports"
    assert verify_prepared_version(config, repository_root=tmp_path)["passed"] is True


def test_version_bundle_rejects_changed_source_or_prepared_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "artifacts" / "runtime" / "detector.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact manifest mismatch"):
        prepare_version_bundle(config, repository_root=tmp_path)

    config = _config(tmp_path / "second")
    prepare_version_bundle(config, repository_root=tmp_path / "second")
    prepared = (
        tmp_path
        / "second"
        / "artifacts"
        / "versions"
        / "0.0.1"
        / "staging"
        / "catalog"
        / "supports.bin"
    )
    prepared.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest mismatch"):
        verify_prepared_version(config, repository_root=tmp_path / "second")


def test_version_bundle_verifies_the_complete_windows_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prepare_version_bundle(config, repository_root=tmp_path)
    version_root = tmp_path / "artifacts" / "versions" / "0.0.1"
    staging = version_root / "staging"
    bundle = version_root / "bixolon-scanner-0.0.1"
    worker = bundle / "worker"
    shutil.copytree(staging / "runtime", worker / "model-package")
    shutil.copytree(staging / "catalog", worker / "store-catalog")
    shutil.copytree(staging / "cuda-runtime", worker / "cuda-runtime")
    (worker / "bixolon-worker.exe").write_bytes(b"worker")
    shutil.copy2(staging / "version.json", bundle / "version.json")
    shutil.copy2(staging / "provenance.json", bundle / "provenance.json")
    with pytest.raises(ValueError, match="product_scanner.exe"):
        write_final_bundle_manifest(config, repository_root=tmp_path, bundle=bundle)
    (bundle / "product_scanner.exe").write_bytes(b"app")

    written = write_final_bundle_manifest(config, repository_root=tmp_path, bundle=bundle)

    result = verify_prepared_version(config, repository_root=tmp_path)

    assert written["file_count"] > 0
    assert result["bundle_path"] == bundle.as_posix()
    assert result["bundle_manifest_sha256"] == sha256_file(bundle / "bundle-manifest.json")
    with pytest.raises(ValueError, match="already exists"):
        write_final_bundle_manifest(config, repository_root=tmp_path, bundle=bundle)
    (worker / "model-package" / "detector.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="final bundle manifest mismatch"):
        verify_prepared_version(config, repository_root=tmp_path)
